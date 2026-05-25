# 프로젝트 설명

(이미지)

사용자가 이미지를 3장 업로드하면,
이미지 기반 감정 분석을 통해 AI 배경음을 생성하고
이미지와 배경음을 활용한 손 인터랙티브 전시를 즐길 수 있는 서비스입니다.

추억하고 싶은 순간을 단순히 보는 것이 아니라,
소리와 인터랙션을 함께 활용하여 실제로 느낄 수 있도록 만들고자 했습니다.

# Backend Focus

(이미지)

백엔드에서는 AI 기반 배경음 생성이 핵심 기능이었기 때문에,
다수의 AI 모델들을 안정적으로 서빙하는 구조 설계에 집중했습니다.

특히

- 장시간 AI inference 처리
- 다중 AI 모델 운영
- 실시간 진행 상태 전달
- Worker 장애 대응
- 제한된 GPU 환경 최적화

등의 문제를 해결하기 위해
Celery + Redis 기반 비동기 파이프라인 구조를 적용했습니다.

# Technical Challenges

## Celery Worker 장애 대응

AI inference 도중 Worker 장애 발생 시 task 유실 및 duplicate execution 가능성이 존재했습니다. 이를 해결하기 위해 late ACK 기반 재처리 구조와 task_id 기반 idempotency 검증 로직을 적용하여 안정성을 개선했습니다.

→ [Blog](#)

## Redis Pub/Sub 메시지 유실 대응

Redis Pub/Sub 기반 SSE 통신 환경에서는 subscribe 시점과 publish 시점 사이의 timing issue로 인해 메시지 유실 가능성이 존재했습니다.
이를 해결하기 위해 MongoDB 기반 fallback 상태 저장 구조와 reconnect 이후 상태 복구 로직을 적용했습니다.

→ [Blog](#)

## GPU 메모리 병목 해결

VRAM 8GB 환경에서 다수의 AI 모델을 동시에 적재할 경우 GPU 메모리 사용량이 급격히 증가하는 문제가 발생했습니다.
이를 해결하기 위해 selective model loading 및 dynamic loading 구조를 적용하여 메모리 사용량을 안정화했습니다.

→ [Blog](#)
