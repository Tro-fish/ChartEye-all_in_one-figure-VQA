import logo from './assets/images/logo.svg';
import './css/App.css';
import './css/fileupload.css';
import './css/progressbar.css';
import './css/extractimg.css';
import './css/dataanalysis.css';
import './css/chat.css';


import { useEffect, useState } from 'react';
import {BrowserRouter as Router, Route, Routes, useNavigate, useLocation} from 'react-router-dom';


import UploadPage from './pages/UploadPage';
import ExtractImage from './pages/ExtractImage';
import DataAnalysis from './pages/DataAnalysis';
import ProgressBar from './components/ProgressBar';

function App() {
  const [step, setStep] = useState(1);
  const [extImages, setExtImages] = useState([]);
  const location = useLocation();
  const navigate = useNavigate();
  const handleNextStep = (images) => {
    setStep((prevStep) => prevStep + 1);
    setExtImages(images)
  };
  useEffect(()=>{
    navigate('/');
  },[]);
  useEffect(() => {
    // 초기 페이지 로드 시 위치에 따라 step 설정
    switch (location.pathname) {
      case '/':
        setStep(1);
        break;
      case '/step2':
        setStep(2);
        break;
      case '/step3':
        setStep(3);
          break;
      default:
        // 기본적으로 첫 페이지로 리다이렉트
        navigate('/');
    }
  }, [location.pathname]); 

  useEffect(() => {
    // step 변경 시 라우트 이동
    switch (step) {
      case 1:
        if (location.pathname !== '/') navigate('/');
        break;
      case 2:
        if (location.pathname !== '/step2') navigate('/step2');
        break;
      case 3:
        if (location.pathname !== '/step3') navigate('/step3');
        break;
    }
  }, [step]);

  return (
      <div className="App">
          <div className='logo-container'>
            <p className='logo-corning'>CORNING</p>
            <div className='logo-charteye'>
              <p>ChartEye</p>
              <img src={'resources/icons/logo.svg'} alt="icon"/>
            </div>
          </div>
        <div className='main-content'>
          <div className='progress-container'>
            <ProgressBar step={step}/>
          </div>
          <Routes>
            <Route exact path="/" element={<UploadPage onNext={handleNextStep}/>}/>
            <Route path="/step2" element={<ExtractImage onNext={handleNextStep} extImages={extImages}/>}/>
            <Route path ="/step3" element={<DataAnalysis/>}/>
          </Routes>
        </div>
      </div>
  );
}

export default App;
