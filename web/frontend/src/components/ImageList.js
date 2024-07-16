import React from 'react';

const ImageList = ({ images, selectedImage, onSelectImage }) => {
  return (
    <div className='image-list-container'>
      {images.map((image, index) => (
        <img 
          key={index}
          src={image}
          alt={`Thumbnail ${index}`}
          className={`thumbnail ${selectedImage === image ? 'selected' : ''}`}
          onClick={() => {onSelectImage(image)}}
        />
      ))}
    </div>
  );
};

export default ImageList;