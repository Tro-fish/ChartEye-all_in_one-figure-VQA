import React, { useState, useEffect } from 'react'

import ImageList from '../components/ImageList';
import { useLocation } from 'react-router-dom';
import Chat from '../components/Chat';

const DataAnalysis = () => {
  const location = useLocation();
  const images = location.state?.images;
  const initialImage = location.state?.selectedImage;
  const [selectedCaption, setSelectedCaption] = useState(location.state?.caption)
  const [selectedImage, setSelectedImage] = useState(initialImage);

  const onImage = async (image) => {
    setSelectedImage(image)
    setSelectedCaption('Loading Caption...')
    const img = selectedImage.replace('data:image/jpeg;base64,', '')

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
        console.error('Error fetching images:', error)
    })
  }

  return (
    <div className="data-analysis-container">
    {images && <ImageList images={images} selectedImage={selectedImage} onSelectImage={onImage} />}
    <div className="analysis-content-container">
      <div className="image-display-container">
        {selectedImage && (
          <>
            <img src={selectedImage} alt="Selected" className="selected-large-image" />
            <div className="image-description">{selectedCaption}</div>
          </>
        )}
      </div>
      <div className="chat-system-container">
        <div className='chat-title'>
          ChartEye AI
        </div>
        <Chat/>
      </div>
    </div>
  </div>
  )
}

export default DataAnalysis