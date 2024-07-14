import React, { useState } from 'react';

const ChatSystem = () => {
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState('');

  const handleInputChange = (e) => {
    setInputText(e.target.value);
  };

  const handleSendClick = () => {
    if (inputText.trim() !== '') {
        sendNewMessage();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && inputText.trim() !== '') {
        e.preventDefault();
        sendNewMessage();
    }
  };

  const sendNewMessage = () => {
    const newMessage = { text: inputText, sender: 'user' };
    setMessages([...messages, newMessage]);
    setInputText('');
    fetchAIResponse(inputText);
  };

  const fetchAIResponse = (text) => {
    const aiResponse = { text: "AI's response: " + text, sender: 'ai' };
    setMessages(messages => [...messages, aiResponse]);
  };

  return (
    <div className="chat-system">
      <div className="message-container">
            {messages.map((msg, index) => (
            <div key={index} className={`message ${msg.sender}`}>
                {msg.text}
            </div>
            ))}
            <div className="input-container">
                <input
                type="text"
                value={inputText}
                onChange={handleInputChange}
                onKeyPress={handleKeyPress} 
                placeholder="질문을 입력하세요"
                />
                <button onClick={handleSendClick}>
                    <span className='material-symbols-outlined'>arrow_upward</span>
                </button>
            </div>
        </div>
    </div>
  );
};

export default ChatSystem;
