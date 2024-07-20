import React, { useState,useEffect, useRef } from 'react';

const ChatSystem = ({image, caption, messages, setMessages}) => {
  const [inputText, setInputText] = useState('');
  const messageEndRef = useRef(null); 

  const scrollToBottom = () => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };
  useEffect(() => {
    scrollToBottom();  // 메시지 목록이 바뀔 때마다 스크롤
  }, [messages]);

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
    const aiMessage = { text: "Thinking...", sender: 'ai', loading: true };
    setMessages(current => [...current, aiMessage]);
    fetchAIResponse(inputText, messages.length+1);

  };


  const fetchAIResponse = async (text, index) => {
    try {
      const response = await fetch('http://127.0.0.1:8000/chat/', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 'image': image, 'text': text, 'caption': caption })
      });
      const data = await response.json();
      setMessages(messages => {
        const updatedMessages = [...messages];
        updatedMessages[index] = { ...updatedMessages[index], text: "<b>Reasoning: </b>" + data.reasoning, loading: false };
        
        const additionalInfo = "<b>Answer: </b>" + data.answer;
        const additionalMessage = { text: additionalInfo, sender: 'ai', loading: false, isAnswer:false };
        updatedMessages.splice(index + 1, 0, additionalMessage); // 새 메시지를 배열에 추가
  
        return updatedMessages;
      });
    } catch (error) {
      console.error('Error getting answer:', error);
      setMessages(messages => {
        const updatedMessages = [...messages];
        updatedMessages[index] = { ...updatedMessages[index], text: "Can't get answer.", loading: false };
        return updatedMessages;
      });
    }
  };
  
  return (
    <div className="chat-system">
      <div className="message-container">
            <div className='messages'>
              {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.sender}`}>
                {msg.sender === 'ai' && (index % 3 === 1) && (
                  <>
                  <div className='chart-eye-container'>
                  <img src={msg.loading ? '/resources/icons/eye.gif' : '/resources/icons/eye.svg'} alt={msg.loading ? 'Loading' : 'Eye'} className="ai-icon"/>
                  </div>
                  </>
                )}
                {msg.sender === 'ai' && (index % 3 === 2) && (
                  <>
                  <div className='chart-eye-container' style={{visibility: 'hidden'}}>
                  <img src={msg.loading ? '/resources/icons/eye.gif' : '/resources/icons/eye.svg'} alt={msg.loading ? 'Loading' : 'Eye'} className="ai-icon"/>
                  </div>
                  </>
                )}
                <div className='message-text' dangerouslySetInnerHTML={{ __html: msg.text }} />
              </div>
              ))}
              <div ref={messageEndRef} />
            </div>
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
