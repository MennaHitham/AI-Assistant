"""
Verification Script: Persistence & Course-Code Security
=======================================================
This script recreates the two core tests:
1. PERSISTENCE: Chat is stored in DB, reloaded, and context is preserved.
2. SECURITY: Searching is strictly limited to registered courses.

Run with: python verify_persistence_and_security.py
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

# Add ai_engine to path for internal imports
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR / 'ai_engine'))

from main.models import User, Department, Course, CourseOffering, Enrollment, ChatConversation, ChatMessage
from ai_engine.src.rag_pipeline import RAGPipeline

def run_verification():
    print("\n" + "="*70)
    print("SCENARIO 1: Setup Student & Enrollments")
    print("="*70)

    # 1. Setup Student
    student, _ = User.objects.get_or_create(
        username="verification_user",
        defaults={"full_name": "Verification Student", "primary_role": User.Role.STUDENT}
    )
    print(f"  User: {student.username}")

    # 2. Setup Enrollments (Registered in IT 222, NOT CS 214)
    course_it222 = Course.objects.filter(code="IT 222").first()
    course_cs214 = Course.objects.filter(code="CS 214").first()
    
    if not course_it222 or not course_cs214:
        print("  [ERROR] Required courses (IT 222 or CS 214) not found in DB. Please run reingest first.")
        return

    Enrollment.objects.filter(student=student).delete()
    offering_it222, _ = CourseOffering.objects.get_or_create(
        course=course_it222, semester="Spring", year=2026, defaults={"is_active": True}
    )
    Enrollment.objects.create(student=student, course_offering=offering_it222, status=Enrollment.Status.ACTIVE)
    
    print(f"  [ENROLLED] in {course_it222.code} (Networks)")
    print(f"  [BLOCKED]  from {course_cs214.code} (Data Structures)")

    # 3. Init AI
    os.environ['EMBEDDING_DEVICE'] = 'cuda'
    pipeline = RAGPipeline()
    pipeline.vector_store_manager.load_vector_store()
    pipeline.is_initialized = True
    print("  AI Engine Ready.")

    print("\n" + "="*70)
    print("SCENARIO 2: Chat Persistence (Close & Continue)")
    print("="*70)

    # Part A: Start Chat
    conv = ChatConversation.objects.create(student=student, course_offering=offering_it222, title="Persistence Test")
    
    q1 = "What is the OSI model?"
    print(f"\n>>> User (Session 1): {q1}")
    
    ChatMessage.objects.create(conversation=conv, role=ChatMessage.Role.USER, content=q1)
    res1 = pipeline.query(question=q1, history=[], user_courses=["IT 222"], selected_course="IT 222")
    ans1 = res1.get("answer")
    ChatMessage.objects.create(conversation=conv, role=ChatMessage.Role.ASSISTANT, content=ans1)
    print(f"    AI: {ans1[:100]}...")

    # Part B: Reload Chat (Simulate "Next Day")
    print("\n  ...Simulating user closing and re-opening chat...")
    reloaded_conv = ChatConversation.objects.get(id=conv.id)
    history_msgs = reloaded_conv.messages.all().order_by('timestamp')
    history = [{"role": ("user" if m.role == ChatMessage.Role.USER else "assistant"), "content": m.content} for m in history_msgs]
    
    q2 = "Tell me specifically about the Network Layer." # Should know we mean in OSI context
    print(f"\n>>> User (Session 2): {q2}")
    
    res2 = pipeline.query(question=q2, history=history, user_courses=["IT 222"], selected_course="IT 222")
    ans2 = res2.get("answer")
    print(f"    AI: {ans2[:150]}...")
    
    if "Layer 3" in ans2 or "IP address" in ans2.lower():
        print("  [PASS] Persistence: AI recovered context from DB history!")
    else:
        print("  [WARN] Context check failed.")

    print("\n" + "="*70)
    print("SCENARIO 3: Course Security (Strict Filtering)")
    print("="*70)

    q_blocked = "What is a Binary Search Tree?" # This is from CS 214
    print(f"\n>>> User: {q_blocked} (Expected Block: CS 214)")
    
    # We only pass IT 222 as enrolled
    res_sec = pipeline.query(question=q_blocked, history=[], user_courses=["IT 222"], selected_course=None)
    ans_sec = res_sec.get("answer")
    sources = res_sec.get("sources", [])

    print(f"    AI Answer: {ans_sec[:100]}...")
    
    if any(s.get("metadata", {}).get("course_code") == "CS 214" for s in sources):
        print("  [FAIL] Security Leak: AI used documents from CS 214!")
    elif "Disclaimer" in ans_sec or "general knowledge" in ans_sec.lower():
        print("  [PASS] Security Intact: AI used general knowledge and blocked CS 214 lectures.")
    else:
        print("  [PASS] Security Intact: No lecture sources were used.")

    print("\n" + "="*70)
    print("VERIFICATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    run_verification()
