css = '''
<style>
.chat-container {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}

.chat-message {
    display: flex;
    margin-bottom: 20px;
}

.chat-message.user .message {
    background-color: #f3f3f3;
    color: #333;
    border-top-right-radius: 20px;
    border-bottom-right-radius: 20px;
    border-bottom-left-radius: 20px;
    padding: 15px;
    max-width: 70%;
}

.chat-message.bot .message {
    background-color: #26272F;
    color: white;
    border-top-left-radius: 20px;
    border-bottom-right-radius: 20px;
    border-bottom-left-radius: 20px;
    padding: 15px;
    max-width: 70%;
}

.avatar img {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    object-fit: cover;
    margin-right: 10px;
}

.user-avatar img {
    float: right;
}

.bot-avatar img {
    float: left;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar bot-avatar">
        <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar user-avatar">
        <img src="https://cdn.pixabay.com/photo/2015/10/05/22/37/blank-profile-picture-973460_960_720.png">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

