from models.teacher import TeacherCNN
from models.student import StudentCNN
from benchmarks.size import model_report

teacher = TeacherCNN()
student = StudentCNN()

print("Teacher:")
print(model_report(teacher))

print("\nStudent:")
print(model_report(student))