import React, { useState, useEffect } from 'react'
import ImageList from '../components/ImageList';
import { useNavigate } from 'react-router-dom';

const ExtractImage = ({onNext, images}) => {
    const [selectedImage, setSelectedImage] = useState(images[0]);

    const [loading, setLoading] = useState(false);
    const [fetching, setFetching] = useState(0); 

    const navigate = useNavigate();
    const handleNextClick = async () => {
        setLoading(true);
        
        // setTimeout(() => {
        //     const dummyCaption = "Temp caption Temp caption Temp caption Temp caption Temp caption Temp caption";
        //     navigate('/step3', { state: { selectedImage, images, caption: dummyCaption } });
        //     setLoading(false);
        // }, 5000); 

        // captioning
        const img = selectedImage.replace('data:image/png;base64,', '');
        try {
            let response = await fetch('http://127.0.0.1:8000/caption/', {
                method: 'POST',
                body: img,
                onUploadProgress: progressEvent => {
                    const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setFetching(percent); // Update progress
                }
            });
            let data = await response.json();
            onNext(selectedImage, data.caption);
        } catch (error) {
            console.error('Error:', error);
        } finally {
            setLoading(false);
        }
    };
    


  return (
    <div className='main-content-container'>
        
        <div className='extracted-box'>
            <header style={{ alignSelf: 'flex-start', display: 'flex', alignItems: 'center', justifyContent: 'space-between', width: '100%' }}>
                    <h3>이미지 추출</h3> 
                    <p style={{ color: 'var(--main-color)', fontFamily: 'Pretendard-Regular', fontSize: '12px' }}>총 {images.length}개의 이미지가 추출되었습니다.</p>
                </header>
                {loading ? (
                    <div className='loading-container'>
                        <h3>데이터 처리 중입니다.</h3>
                        <p className='sub-text'>캡션 모델링 중...</p>
                        <p className='percentage-text'>{fetching}%</p>
                        <div className="loading-bar-container">
                            <div className="loading-bar" style={{ width: `${fetching}%` }}></div>
                        </div>
                    </div>
                ) : (
                    <>
                        <div className='extracted-selected-container'>
                        {selectedImage && <img src={selectedImage} alt="Selected"/>}
                        </div>
                        <ImageList images={images} selectedImage={selectedImage} onSelectImage={setSelectedImage} />
                        
                        <button className={`button-next ${images.length > 0 ? '' : 'disabled'}`}
                        style={{ alignSelf: 'flex-end' }} onClick={handleNextClick}>
                            다음</button>
                    </>
                )}
                
        </div>

    </div>
  )
}

export default ExtractImage