import React, { useCallback, useState } from 'react'
import {useDropzone} from 'react-dropzone';
import CircularProgress from '@mui/material/CircularProgress';

function FileUpload ({onFilesAdded}) {
    const [files, setFiles] = useState([]);
    const icons = [
        'resources/icons/pdf.svg',
        'resources/icons/ppt.svg',
        'resources/icons/doc.svg',
        'resources/icons/file.svg'
      ];
    const onDrop = useCallback((acceptedFiles) => {
        const updatedFiles = acceptedFiles.map(file => ({
            ...file,
            file: file,
            preview: URL.createObjectURL(file),
            icon: file.name.toLowerCase().endsWith('.pdf') ? 0 :
                file.name.toLowerCase().endsWith('.ppt') || file.name.toLowerCase().endsWith('.pptx') ? 1 :
                file.name.toLowerCase().endsWith('.doc') || file.name.toLowerCase().endsWith('.docx') ? 2:
                3,
            loading: true,
        }));

        setFiles(prevFiles => [...prevFiles, ...updatedFiles]); 
        onFilesAdded(updatedFiles);

        setTimeout(()=>{
            setFiles(prevFiles => prevFiles.map(file=>({
                ...file,
                loading: false
            })));
        },1000);
    },[onFilesAdded]);

    const {getRootProps, getInputProps, isDragActive} = useDropzone({onDrop});

  return (
    <div className='drag-box-container'>
        <div {...getRootProps()} className={`drag-box ${isDragActive ? 'active' : ''}`}>

            <div className='material-symbols-outlined'>move_to_inbox</div>
            <input {...getInputProps()}/>
            {
                isDragActive ?
                <span className='drag-box-main-text'>파일을 여기에 놓으세요</span> :
                <span className='drag-box-main-text'>파일을 여기에 드래그 하거나 클릭하여 업로드하세요</span>
                
            }
            <span className='drag-box-sub-text'>차트가 포함된 Word, pdf, ppt 파일</span>
        </div>
        
        <div className='file-list'>
            {files.map(file=>(
                <div key = {file.path} className='upload-file-container'>
                    <div className='upload-file-icon-container'>
                        {file.loading ? (
                        <CircularProgress size={24} />
                        ) : (<img src={icons[file.icon]} alt="icon"/>)}
                    </div>
                    
                    <p>{file.path}</p>
                </div>
            ))}
        </div>
    </div>
  );
}

export default FileUpload