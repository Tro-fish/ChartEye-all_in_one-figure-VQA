import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';

function StepNavigator({ step }) {
  const navigate = useNavigate();

  useEffect(() => {
    switch (step) {
    case 1:
        navigate('/')
      case 2:
        navigate('/extract-image');
        break;
      // 추가적인 케이스 처리 가능
      default:
        break;
    }
  }, [step, navigate]);

  return null; // UI 렌더링이 필요 없는 경우
}
