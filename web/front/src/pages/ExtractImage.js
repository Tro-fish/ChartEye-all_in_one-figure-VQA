import React, { useState, useEffect } from 'react'
import ImageList from '../components/ImageList';
import { useNavigate } from 'react-router-dom';

const ExtractImage = ({onNext}) => {
    const [images, setImages] = useState([
        'resources/figures/image2_1.png',
        'resources/figures/image2_2.png',
        'resources/figures/image4_7.png',
        'resources/figures/image4_8.png',
        'resources/figures/image4_9.png',
        'resources/figures/image4_10.png',
        'resources/figures/image6_1.png',
        'resources/figures/image9_1.png',
        'resources/figures/image9_4.png',
        'resources/figures/image9_5.png',
    ]);
    const [selectedImage, setSelectedImage] = useState(images[0]);
    const navigate = useNavigate();
    const handleNextClick =() =>{
        navigate('/step3', {state: {selectedImage, images}});
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