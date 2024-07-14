import React, {useState} from 'react'
import FileUpload from '../components/FileUpload'

const UploadPage = ({onNext}) => {
    const [files, setFiles] = useState([]); 
    //const [loading, setLoading] = useState(false);
    const handleNextClick =() =>{
        if(files.length > 0){
            onNext();
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
            <FileUpload onFilesAdded={handleNewFiles}/>
            <button className={`button-next ${files.length > 0 ? '' : 'disabled'}`}
            style={{ alignSelf: 'flex-end' }} onClick={handleNextClick}>
                다음</button>    
        </div>
    </div>
  )
}

export default UploadPage