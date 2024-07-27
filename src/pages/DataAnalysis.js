import React, { useState, useEffect } from 'react'

import ImageList from '../components/ImageList';
import { useLocation } from 'react-router-dom';
import Chat from '../components/Chat';

import CircularProgress from '@mui/material/CircularProgress';


const DataAnalysis = ({images, initialImage, initialCaption}) => {
  const location = useLocation();
  const [selectedCaption, setSelectedCaption] = useState(initialCaption)
  const [selectedImage, setSelectedImage] = useState(initialImage);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState([]);

  const onImage = async (image) => {
    setSelectedImage(image)
    setSelectedCaption('Loading Caption...');
    setLoading(true);
    setMessages([])

    const img = image.replace('data:image/png;base64,', '')

    await fetch('http://127.0.0.1:8000/caption/', {
      method: 'POST',
      body: img
    })
    .then(response => response.json())
    .then(data => {
      const caption = data.caption
      setSelectedCaption(caption)
    })
    .catch(error => {
        setSelectedCaption("Can't load caption.")
        console.error('Error loading caption:', error)
    })
    .finally(() => {
      setLoading(false);
    });
  }

  const onOpenImage = () => {
    var imageWin = new Image();
    imageWin = window.open("", "", "width=2000px, height=1000px");
    imageWin.document.write("<html><body style='margin:0'>");
    imageWin.document.write("<img src='" + selectedImage + "' border=0>");
    imageWin.document.write("</body><html>");
  }

  return (
    <div className="data-analysis-container">
      <div className='analysis-image-list-container'>
      {images && <ImageList images={images} selectedImage={selectedImage} onSelectImage={onImage} />}
      </div>
    <div className="analysis-content-container">
      <div className="image-display-container">
        <h4 className='chat-title'>
          Figure
        </h4>
        {selectedImage && (
          <>
            {loading ? (
                <CircularProgress/>) : (
                <img src={selectedImage} alt="Selected" className="selected-large-image" onClick={onOpenImage} style={{cursor: 'pointer'}} />
              )}
            <div className="image-description">
              <p style={{ alignSelf: 'flex-start', color: 'var(--sub-text-color)', margin: '0' }}>Generated Caption</p>
            {loading ? (
                <>...</>) : (
                <>{selectedCaption}</>
              )}</div>
          </>
        )}
      </div>
      <div className="chat-system-container">
        <h4 className='chat-title'>
          ChartEye AI
        </h4>
        <Chat image={selectedImage} caption={selectedCaption} messages={messages} setMessages={setMessages}/>
      </div>
    </div>
  </div>
  )
}

export default DataAnalysis