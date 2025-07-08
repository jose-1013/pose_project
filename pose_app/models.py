from django.db import models

# 운동 카테고리 선택지
CATEGORY_CHOICES = [
    ("크로스핏", "크로스핏"),
    ("요가", "요가"),
]

# 포즈 타입 선택지 (원한다면 자유 입력도 가능)
POSE_CHOICES = [
    ("squat", "스쿼트"),
    ("cat", "고양이"),
]

class ReferencePose(models.Model):
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES)  # 🟢 요가, 크로스핏
    pose_type = models.CharField(max_length=100, choices=POSE_CHOICES)    # 🟢 squat, cat 등 내부 식별용
    name = models.CharField(max_length=100)                               # 🟢 사용자에게 보일 이름 (예: 고양이 자세)
    csv_path = models.CharField(max_length=200)                           # 🟢 기준자세 파일 경로
    row_index = models.IntegerField()                                     # 🟢 기준 포즈가 위치한 행 번호

    def __str__(self):
        return f"[{self.category}] {self.pose_type} - {self.name}"
