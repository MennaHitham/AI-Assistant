import os
import django
import json

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'GraduationProject.settings')
django.setup()

from main.models import User, ChatConversation, ChatMessage
from student.views import StudentChatBotView, StudentChatConversationDetailView
from rest_framework.test import APIRequestFactory, force_authenticate

def simulate_chat():
    factory = APIRequestFactory()
    user = User.objects.get(username='student001')
    chat_view = StudentChatBotView.as_view()
    detail_view = StudentChatConversationDetailView.as_view()

    print("--- STEP 1: Starting a new conversation ---")
    payload1 = {"content": "What is the main topic of the course?"}
    request1 = factory.post('/api/student/chat/', payload1, format='json')
    force_authenticate(request1, user=user)
    response1 = chat_view(request1)
    
    if response1.status_code == 200:
        convo_id = response1.data['conversation_id']
        print(f"AI Response 1 (Convo ID: {convo_id}):")
        print(response1.data['ai_message']['content'])
    else:
        print(f"Error in Step 1: {response1.data}")
        return

    print("\n--- STEP 2: Listing all conversations ---")
    request2 = factory.get('/api/student/chat/')
    force_authenticate(request2, user=user)
    response2 = chat_view(request2)
    if response2.status_code == 200:
        print(f"Found {len(response2.data)} conversations.")
        for c in response2.data:
            print(f"- ID: {c['id']}, Title: {c['title']}, Last: {c['last_message']}")
    else:
        print(f"Error in Step 2: {response2.data}")

    print(f"\n--- STEP 3: Continuing conversation {convo_id} ---")
    payload3 = {"content": "Can you summarize the most important point from your last answer?", "conversation_id": convo_id}
    request3 = factory.post('/api/student/chat/', payload3, format='json')
    force_authenticate(request3, user=user)
    response3 = chat_view(request3)

    if response3.status_code == 200:
        print("AI Response 3 (Context Aware):")
        print(response3.data['ai_message']['content'])
    else:
        print(f"Error in Step 3: {response3.data}")

    print(f"\n--- STEP 4: Retrieving history for conversation {convo_id} ---")
    request4 = factory.get(f'/api/student/chat/{convo_id}/')
    force_authenticate(request4, user=user)
    response4 = detail_view(request4, pk=convo_id)
    if response4.status_code == 200:
        messages = response4.data['messages']
        print(f"Retrieved {len(messages)} messages for conversation {convo_id}.")
        for m in messages:
            print(f"  [{m['role']}] {m['content'][:100]}...")
        
        if len(messages) >= 4:
             print("\n[SUCCESS] Chat history is correctly saved and retrievable.")
        else:
             print("\n[FAILURE] Expected at least 4 messages (2 user, 2 AI).")
    else:
        print(f"Error in Step 4: {response4.data}")

if __name__ == "__main__":
    simulate_chat()
