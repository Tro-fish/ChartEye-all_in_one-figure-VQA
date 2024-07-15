import React, { useState, useEffect } from 'react'

import ImageList from '../components/ImageList';
import { useLocation } from 'react-router-dom';
import Chat from '../components/Chat';

const DataAnalysis = () => {
  const location = useLocation();
  const images = location.state?.images;
  const initialImage = location.state?.selectedImage;
  const [selectedImage, setSelectedImage] = useState(initialImage);
  return (
    <div className="data-analysis-container">
    {images && <ImageList images={images} selectedImage={selectedImage} onSelectImage={setSelectedImage} />}
    <div className="analysis-content-container">
      <div className="image-display-container">
        {selectedImage && (
          <>
            <img src={selectedImage} alt="Selected" className="selected-large-image" />
            <div className="image-description">Image description here...</div>
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