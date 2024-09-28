from django.db import models

class UserFeedback(models.Model):
    user_name = models.CharField(max_length=100, null=True, blank=True)  # Store user's name
    user_input = models.TextField()
    response = models.TextField()
    feedback = models.CharField(max_length=10)  # e.g., "yes" or "no"
    feedback_context = models.TextField(null=True, blank=True)  # Store context of feedback
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback from {self.user_name or 'Anonymous'} - {self.feedback}"
