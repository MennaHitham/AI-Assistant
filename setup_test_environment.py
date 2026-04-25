import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'GraduationProject.settings')
django.setup()

from main.models import User, Course, CourseOffering, Department, Enrollment

def setup():
    # 1. Update student001 role
    try:
        student = User.objects.get(username='student001')
        student.primary_role = User.Role.STUDENT
        student.full_name = "Test Student"
        student.save()
        print(f"[OK] Updated {student.username} to STUDENT role.")
    except User.DoesNotExist:
        print("! Error: student001 not found.")
        return

    # 2. Create a Department if none exists
    dept, _ = Department.objects.get_or_create(
        code="CS",
        defaults={"name": "Computer Science"}
    )
    print(f"[OK] Department: {dept.name}")

    # 3. Create a Course
    course, _ = Course.objects.get_or_create(
        code="CS101",
        defaults={
            "name": "Intro to Programming",
            "department": dept,
            "credit_hours": 3
        }
    )
    print(f"[OK] Course: {course.name}")

    # 4. Create a Course Offering
    offering, _ = CourseOffering.objects.get_or_create(
        course=course,
        semester="Spring",
        year=2025,
        defaults={"capacity": 50}
    )
    print(f"[OK] Course Offering: {offering}")

    # 5. Enroll the Student
    enrollment, created = Enrollment.objects.get_or_create(
        student=student,
        course_offering=offering,
        defaults={"status": Enrollment.Status.ACTIVE}
    )
    if created:
        print("[OK] Enrolled student001 in CS101.")
    else:
        print("[OK] Student already enrolled.")

if __name__ == "__main__":
    setup()
