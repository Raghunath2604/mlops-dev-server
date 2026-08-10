import resend
import os
from pathlib import Path

resend.api_key = os.environ.get("RESEND_API_KEY", "")

def send_waitlist_confirmation(email: str, name: str, position: int):
    if not resend.api_key:
        print(f"Skipping email to {email} - RESEND_API_KEY not set")
        return
        
    try:
        # Load the HTML template
        template_path = Path(__file__).parent.parent.parent / "email" / "waitlist-confirmation.html"
        if template_path.exists():
            template = template_path.read_text(encoding='utf-8')
        else:
            template = "Hi {{NAME}}, you're #{{POSITION}} on the waitlist!"
            
        html = template.replace("{{POSITION}}", f"{position}").replace("{{NAME}}", name or "there")

        params = {
            "from": "Raghunathareddy GR <hello@mlopsde.me>",
            "to": [email],
            "subject": f"You're #{position} on the MLOps.dev waitlist",
            "html": html,
        }
        return resend.Emails.send(params)
    except Exception as e:
        print(f"Failed to send email to {email}: {e}")
