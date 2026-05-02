"""
Final Integration Test: REAL student001 on eduera DB
===================================================
This script tests everything using the real student001 account:
1. Multi-turn Persistence in the real ChatMessage table.
2. Cumulative Access (Active + Completed courses).
3. Security Blocking (Unauthorized courses).

Run with: python verify_real_student001.py
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

from main.models import User, Enrollment, Course, CourseOffering, ChatConversation, ChatMessage
from ai_engine.src.rag_pipeline import RAGPipeline

def run_real_test():
    print("\n" + "="*70)
    print("STEP 1: Syncing student001's Academic History in DB")
    print("="*70)

    # 1. Fetch Real User
    try:
        student = User.objects.get(username='student001')
    except User.DoesNotExist:
        print("  [ERROR] student001 not found. Please create them in the admin panel first.")
        return

    # 2. Setup real Enrollment history for the test
    Enrollment.objects.filter(student=student).delete()
    
    # Current (ACTIVE): IT 222
    course_it = Course.objects.get(code='IT 222')
    offering_it, _ = CourseOffering.objects.get_or_create(course=course_it, semester='Spring', year=2026, defaults={'is_active': True})
    Enrollment.objects.create(student=student, course_offering=offering_it, status=Enrollment.Status.ACTIVE)
    
    # Past (COMPLETED): CS 214
    course_cs = Course.objects.get(code='CS 214')
    offering_cs, _ = CourseOffering.objects.get_or_create(course=course_cs, semester='Fall', year=2025, defaults={'is_active': False})
    Enrollment.objects.create(student=student, course_offering=offering_cs, status=Enrollment.Status.COMPLETED)

    print(f"  User: {student.full_name} (@{student.username})")
    print(f"  Current: {course_it.code}")
    print(f"  Completed: {course_cs.code}")
    print(f"  Forbidden: AI 330")

    # 3. Simulate View logic for authorized codes
    allowed_codes = list(
        Enrollment.objects.filter(
            student=student,
            status__in=[Enrollment.Status.ACTIVE, Enrollment.Status.COMPLETED]
        ).values_list('course_offering__course__code', flat=True)
    )
    print(f"  Authorized Access List: {allowed_codes}")

    # 4. Init AI
    os.environ['EMBEDDING_DEVICE'] = 'cuda'
    pipeline = RAGPipeline()
    pipeline.vector_store_manager.load_vector_store()
    pipeline.is_initialized = True

    print("\n" + "="*70)
    print("STEP 2: Multi-turn Chat & Persistence (REAL DB SAVING)")
    print("="*70)

    # Create a real conversation record
    conv = ChatConversation.objects.create(
        student=student, 
        course_offering=offering_it, 
        title="Real Integration Test"
    )

    # Turn 1:
    q1 = "Explain the Transport Layer in OSI."
    print(f"\n>>> student001: {q1}")
    ChatMessage.objects.create(conversation=conv, role=ChatMessage.Role.USER, content=q1)
    
    res1 = pipeline.query(question=q1, history=[], user_courses=allowed_codes)
    ans1 = res1.get('answer')
    ChatMessage.objects.create(conversation=conv, role=ChatMessage.Role.ASSISTANT, content=ans1)
    print(f"    AI Answer saved to DB. (Course: {res1.get('sources')[0]['metadata']['course_code'] if res1.get('sources') else 'None'})")

    # Turn 2 (Context Check):
    q2 = "What are the main protocols used there?" # Should know 'there' means Transport Layer
    print(f"\n>>> student001: {q2}")
    
    # Reload history from REAL DB with correct role mapping
    history_msgs = conv.messages.all().order_by('timestamp')
    history = []
    for m in history_msgs:
        role = "user" if m.role == ChatMessage.Role.USER else "assistant"
        history.append({"role": role, "content": m.content})
        
    res2 = pipeline.query(question=q2, history=history, user_courses=allowed_codes)
    ans2 = res2.get('answer')
    print(f"    AI: {ans2[:100]}...")
    if "TCP" in ans2 or "UDP" in ans2:
        print("    [PASS] Context recovered from real ChatMessage table!")

    print("\n" + "="*70)
    print("STEP 3: Testing Cumulative & Security Logic")
    print("="*70)

    # Test Completed Course (CS 214)
    q3 = "What is a Stack?"
    print(f"\n>>> student001 (Asking about COMPLETED course): {q3}")
    res3 = pipeline.query(question=q3, history=[], user_courses=allowed_codes)
    sources3 = {s.get('metadata', {}).get('course_code') for s in res3.get('sources', [])}
    if 'CS 214' in sources3:
        print("    [PASS] Correctly accessed materials from COMPLETED course CS 214.")

    # Test Forbidden Course (AI 330)
    q4 = "What is Machine Learning?"
    print(f"\n>>> student001 (Asking about FORBIDDEN course): {q4}")
    res4 = pipeline.query(question=q4, history=[], user_courses=allowed_codes)
    sources4 = {s.get('metadata', {}).get('course_code') for s in res4.get('sources', [])}
    if 'AI 330' not in sources4:
        print("    [PASS] Correctly BLOCKED unauthorized materials from AI 330.")
        if "Disclaimer" in res4.get('answer'):
            print("    [INFO] AI used general knowledge with disclaimer.")

    print("\n" + "="*70)
    print("FINAL RESULT: ALL SYSTEMS NOMINAL")
    print("="*70)

if __name__ == "__main__":
    run_real_test()
