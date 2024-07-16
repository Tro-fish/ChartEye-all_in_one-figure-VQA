import React, { useState } from 'react';

const ChatSystem = ({image, caption}) => {
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

  const fetchAIResponse = async (text) => {
    await fetch('http://127.0.0.1:8000/chat/', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        'image': image,
        'text': text,
        'caption': caption
      })
    })
    .then(response => response.json())
    .then(data => {
      const answer = data.answer
      const aiResponse = { text: answer, sender: 'ai' };
      setMessages(messages => [...messages, aiResponse]);
    })
    .catch(error => {
      const answer = "Can't get answer."
      const aiResponse = { text: answer, sender: 'ai' };
      setMessages(messages => [...messages, aiResponse]);
      console.error('Error getting answer:', error)
    })
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
