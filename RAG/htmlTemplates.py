# htmlTemplates.py

css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 1rem; 
    margin-bottom: 1.5rem; 
    display: flex; 
    align-items: center; 
    transition: all 0.3s ease;
    box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
}
.chat-message.user {
    background-color: #2b313e;
    border-left: 5px solid #00bfa5;
}
.chat-message.bot {
    background-color: #475063;
    border-left: 5px solid #007bff;
}
.chat-message .avatar img {
    max-width: 60px; 
    max-height: 60px; 
    border-radius: 50%;
    object-fit: cover; 
}
.chat-message .message {
    color: #fff;
    font-size: 1rem;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.freepik.com/256/4712/4712109.png" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://uxwing.com/wp-content/themes/uxwing/download/peoples-avatars/corporate-user-icon.png" alt="User Avatar">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''
