import React, {useState} from 'react'
import FileUpload from '../components/FileUpload'

const UploadPage = ({onNext}) => {
    const [files, setFiles] = useState([]); 
    //const [loading, setLoading] = useState(false);
    
    const handleNextClick = async () =>{
        if(files.length > 0){
            const formData = new FormData()
            files.forEach((file, index) => {
                formData.append(`file${index}`, file.file)
            })
            console.log(formData)

            await fetch('http://127.0.0.1:8000/extract/', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                onNext(data.images)
            })
            .catch(error => {
                console.error('Error fetching images:', error)
            })
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