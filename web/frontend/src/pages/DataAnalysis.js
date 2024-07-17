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

  const onImage = async (image) => {
    setSelectedImage(image)
    setSelectedCaption('Loading Caption...');
    setLoading(true);

    const img = image.replace('data:image/png;base64,', '')

    await fetch('http://127.0.0.1:8000/caption/', {
        method: 'POST',
        body: img
    })
    .then(response => response.json())
    .then(data => {
      if (1 === 1) { // 현재 요청에 대한 응답인지 확인
        const caption = data.caption
        setSelectedCaption(caption)
      }
    })
    .catch(error => {
        setSelectedCaption("Can't load caption.")
        console.error('Error loading caption:', error)
    })
    .finally(() => {
      setLoading(false);
    });
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
                <img src={selectedImage} alt="Selected" className="selected-large-image" />
              )}
            <div className="image-description">
              <p style={{ alignSelf: 'flex-start', color: 'var(--sub-text-color)', margin: '0' }}>추출된 캡션 데이터</p>
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
        <Chat image={selectedImage} caption={selectedCaption}/>
      </div>
    </div>
  </div>
  )
}

export default DataAnalysis