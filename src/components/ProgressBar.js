import React from 'react'

function ProgressBar ({step}) {
  return (
    <div className="progress-bar">
        <ul>
            <li className={step>=1? 'active':''}>
              <div className='icon-container'>
                <span className='material-symbols-outlined'>move_to_inbox</span>
              </div>
              <span className='step-number'>1 단계</span> 
              <span className='step-text'>파일 업로드</span>
            </li>
            
            <li className={step>=2? 'active':''}>
              <div className='icon-container'>
                <span className='material-symbols-outlined'>photo_library</span>
              </div>
              <span className='step-number'>2 단계</span> 
              <span className='step-text'>이미지 추출</span>
            </li>
            <li className={step>=3? 'active':''}>
              <div className='icon-container'>
                <span className='material-symbols-outlined'>analytics</span>
              </div>
              <span className='step-number'>3 단계</span> 
              <span className='step-text'>데이터 분석</span>
            </li>
        </ul>
    </div>
  )
}

export default ProgressBar