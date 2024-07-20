import React, {useState} from 'react';
import FileUpload from '../components/FileUpload';

const UploadPage = ({onNext}) => {
    const [files, setFiles] = useState([]); 
    const [loading, setLoading] = useState(false);
    const [fetching, setFetching] = useState(0);
    
    const handleNextClick = async () =>{
        if(files.length > 0){
            setLoading(true);
            const formData = new FormData()
            files.forEach((file, index) => {
                formData.append(`file${index}`, file.file)
            })
        
            const options = {
                method: 'POST',
                body: formData
            };

            await fetch('http://127.0.0.1:8000/extract/', options, {
                onUploadProgress: progressEvent => {
                    const percent = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    setFetching(percent);
                }
            })

            .then(response => response.json())
            .then(data => {
                onNext(data.images);
                setLoading(false);
            })
            .catch(error => {
                console.error('Error fetching images:', error);
                setLoading(false);
            });
        }
    };

    const handleNewFiles = (newFiles) => {
        setFiles(prevFiles => [...prevFiles, ...newFiles]);
    };
  return (
    <div className='main-content-container'>
        <div className='upload-box'>
            
            <header style={{ alignSelf: 'flex-start' }}>
                <h3>파일 업로드</h3>
            </header>
            {loading ? ( 
                <div className='loading-container'>
                    <h3>데이터 처리 중입니다.</h3>
                    <p className='sub-text'>Figure 이미지 추출 중 ...</p>
                <div className="loading-bar-container">
                    
                    <div className="loading-bar-extract"></div>
                </div>
                </div>
            )
            :
            (<>
                <FileUpload onFilesAdded={handleNewFiles}/>
                <button className={`button-next ${files.length > 0 ? '' : 'disabled'}`}
                style={{ alignSelf: 'flex-end' }} onClick={handleNextClick}>
                    다음</button>
            </>)}
            
        </div>
    </div>
  )
}

export default UploadPage