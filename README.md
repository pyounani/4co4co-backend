

## 기억의 차원

`#이미지 기반 AI 배경음 생성` `#인터랙티브 전시 경험` <br /> <br />
사용자가 이미지를 3장 업로드하면, 이미지 기반 감정 분석을 통해 AI 배경음을 생성하고 이미지와 배경음으로 인터랙티브 전시를 즐길 수 있는 서비스입니다. 추억하고 싶은 순간을 단순히 보는 것이 아니라, 소리와 인터랙션을 함께 활용하여 실제로 느낄 수 있도록 만들고자 했습니다.

> 2025 캡스톤디자인 졸업 프로젝트 <br />
> 개발 기간: 2025.03 ~ 2025.10

 
<img width="400" height="225" alt="시연영상_제출본 (1)" src="https://github.com/user-attachments/assets/3ca32308-daf6-45a7-9ed6-ecf00a146cc8" /> <br />
[Go to YouTube Link](https://www.youtube.com/watch?v=WWsEPMtPJT4)


<br />



## Backend Focus
백엔드에서는 AI 기반 배경음 생성이 핵심 기능이었기 때문에,
다수의 AI 모델들을 안정적으로 서빙하는 구조 설계에 집중했습니다.

- 장시간 AI inference 처리
- 다중 AI 모델 운영
- 실시간 진행 상태 전달
- Worker 장애 대응
- 제한된 GPU 환경 최적화

<br />

### Technical Challenges

#### 1. Celery Worker 장애 대응
- Celery Worker 장애로 인한 task ACK 이전 유실 및 중복 실행 문제를 확인하고, late ACK 및 idempotency 기반 재처리 구조를 적용해 안정성을 개선했습니다.

→ 자세한 분석은 [Blog](https://pyounani.tistory.com/35) 참고 
<br /> 
<br />

#### 2. Redis Pub/Sub 메시지 유실 대응
- Redis Pub/Sub 기반 SSE 통신 환경에서 구독 시점 차이로 발생할 수 있는 메시지 유실 문제를 분석하고, MongoDB 기반 Fallback 로직을 통해 안정성을 개선했습니다.

→ 자세한 분석은 [Blog](https://pyounani.tistory.com/36) 참고 
<br /> 
<br />

#### 3. GPU 메모리 병목 해결
- VRAM 8GB 환경에서 다중 AI 모델 서빙 시 발생하는 메모리 병목 문제를 분석하고, 입력 기반 모델 동적 로딩 구조를 통해 GPU 사용량을  안정화했습니다.

→ 자세한 분석은 [Blog](https://pyounani.tistory.com/37) 참고 
<br /> 


<br />

### Service Architecture

<img width="1561" height="867" alt="image" src="https://github.com/user-attachments/assets/8b249312-318b-432b-b241-470b1d97ad15" />
