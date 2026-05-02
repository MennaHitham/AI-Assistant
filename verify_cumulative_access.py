"""
Verification Script: Cumulative Course Access (Active + Completed)
==================================================================
This script verifies that students can access materials from:
1. Current term courses (ACTIVE)
2. Previous term courses (COMPLETED)
But are still BLOCKED from courses they never registered for.

Run with: python verify_cumulative_access.py
"""
import os
import sys
import django
from pathlib import Path

# Fix Windows encoding
sys.stdout.reconfigure(encoding='utf-8')

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'GraduationProject.settings')
django.setup()

# Add ai_engine to path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / 'ai_engine'))

from main.models import User, Department, Course, CourseOffering, Enrollment
from ai_engine.src.rag_pipeline import RAGPipeline

def run_verification():
    print("\n" + "="*70)
    print("STEP 1: Setting up Cumulative Enrollment Scenario")
    print("="*70)

    # 1. Setup Student
    student, _ = User.objects.get_or_create(
        username="cumulative_user",
        defaults={"full_name": "Cumulative Test Student", "primary_role": User.Role.STUDENT}
    )
    Enrollment.objects.filter(student=student).delete()

    # 2. Setup Courses
    course_active = Course.objects.filter(code="IT 222").first()   # Networks (Active)
    course_completed = Course.objects.filter(code="CS 214").first() # Data Structure (Completed)
    course_blocked = Course.objects.filter(code="AI 330").first()   # Machine Learning (Never Registered)

    if not all([course_active, course_completed, course_blocked]):
        print("  [ERROR] Required courses not found. Please run reingest first.")
        return

    # Create Enrollments
    # Active: IT 222
    offering_active, _ = CourseOffering.objects.get_or_create(
        course=course_active, semester="Spring", year=2026, defaults={"is_active": True}
    )
    Enrollment.objects.create(student=student, course_offering=offering_active, status=Enrollment.Status.ACTIVE)
    
    # Completed: CS 214
    offering_completed, _ = CourseOffering.objects.get_or_create(
        course=course_completed, semester="Fall", year=2025, defaults={"is_active": False}
    )
    Enrollment.objects.create(student=student, course_offering=offering_completed, status=Enrollment.Status.COMPLETED)

    print(f"  [STATUS] {course_active.code} -> ACTIVE")
    print(f"  [STATUS] {course_completed.code} -> COMPLETED")
    print(f"  [STATUS] {course_blocked.code} -> NEVER REGISTERED")

    # Fetch what the VIEW would see
    enrolled_codes = list(
        Enrollment.objects.filter(
            student=student,
            status__in=[Enrollment.Status.ACTIVE, Enrollment.Status.COMPLETED],
        ).select_related('course_offering__course')
        .values_list('course_offering__course__code', flat=True)
    )
    print(f"  [VIEW OUTPUT] Allowed codes: {enrolled_codes}")

    # 3. Init AI
    os.environ['EMBEDDING_DEVICE'] = 'cuda'
    pipeline = RAGPipeline()
    pipeline.vector_store_manager.load_vector_store()
    pipeline.is_initialized = True
    print("  AI Engine Ready.")

    print("\n" + "="*70)
    print("STEP 2: Testing Access Logic")
    print("="*70)

    test_cases = [
        ("What is the OSI model?", "IT 222", "ACTIVE"),
        ("What is a stack in data structures?", "CS 214", "COMPLETED"),
        ("What is Linear Regression?", "AI 330", "BLOCKED")
    ]

    for q, expected_code, category in test_cases:
        print(f"\n>>> Query ({category}): {q}")
        res = pipeline.query(question=q, history=[], user_courses=enrolled_codes, selected_course=None)
        ans = res.get("answer")
        sources = res.get("sources", [])
        
        found_sources = {s.get("metadata", {}).get("course_code") for s in sources}
        print(f"    Sources found: {found_sources if found_sources else 'None'}")
        
        if category == "BLOCKED":
            if expected_code in found_sources:
                print(f"    [FAIL] Security Leak! AI used materials from blocked course {expected_code}")
            else:
                print(f"    [PASS] Correctly blocked materials from {expected_code}")
        else:
            if expected_code in found_sources:
                print(f"    [PASS] Successfully retrieved materials from {category} course {expected_code}")
            else:
                print(f"    [WARN] No materials found for {expected_code}, but access was granted.")

    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_verification()
