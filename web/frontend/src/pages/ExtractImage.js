import React, { useState, useEffect } from 'react'
import ImageList from '../components/ImageList';
import { useNavigate } from 'react-router-dom';

const ExtractImage = ({onNext, extImages}) => {
    const images = extImages.map(img => `data:image/png;base64,${img}`)
    const [selectedImage, setSelectedImage] = useState(images[0]);
    const navigate = useNavigate();
    const handleNextClick = async () => {
        const img = selectedImage.replace('data:image/png;base64,', '')

        // captioning
        await fetch('http://127.0.0.1:8000/caption/', {
            method: 'POST',
            body: img
        })
        .then(response => response.json())
        .then(data => {
            const caption = data.caption
            navigate('/step3', {state: {selectedImage, images, caption}});
        })
        .catch(error => {
            console.error('Error fetching images:', error)
        })
    };
    


  return (
    <div className='main-content-container'>
        
        <div className='extracted-box'>
            <header style={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', justifyContent: 'space-between',width: '100%' }}>
                    <h3>이미지 추출</h3> 
                    <p style={{ color: 'var(--main-color)', fontFamily: 'Pretendard-Regular',fontSize:'12px' }}>총 {images.length}개의 이미지가 추출되었습니다.</p>
            </header>
                <div className='extracted-selected-container'>
                    {selectedImage && <img src={selectedImage} alt="Selected"/>}
                </div>

                <ImageList images={images} selectedImage={selectedImage} onSelectImage={setSelectedImage} />
                <button className={`button-next ${(true) > 0 ? '' : 'disabled'}`}
                style={{ alignSelf: 'flex-end' }} onClick={handleNextClick}>
                    다음</button>   
        </div>
    </div>
  )
}

export default ExtractImage