<p align="center">
  <img src="resources/icon.png" width="88" alt="DFRAGON 빨간 용" />
</p>

<h1 align="center">Real OCR Evaluation</h1>

<p align="center">
  캡처 데이터를 나누고, <strong>OCR 정확도와 오답을 확인하는 Windows 앱</strong>입니다.
</p>

<p align="center">
  <a href="https://github.com/blahaj94/dfragon-ocr-eval-tool/releases/latest"><strong>Windows 다운로드</strong></a> · 설치 파일 또는 ZIP
</p>

## 01. 모델과 데이터 선택

**학습** 탭에서 미니PC의 자료실에 로그인하고 모델·train/val/test 데이터를 가져와 Windows GPU로 학습한 뒤 평가할 수 있습니다. 실제 train을 유지하면서 목표 문자 분포의 부족분을 합성으로 보충하고, 선택한 새 문자를 추가한 모델도 학습할 수 있습니다. 새 모델은 **미니PC에 올리기**로 등록합니다.

<p align="center">
  <img src="docs/screenshots/setup.png" width="352" alt="실제 앱에서 Python, 학습 결과, 체크포인트, Cropper 캡처와 정답 파일을 선택한 화면" />
</p>

## 02. 평가 실행 → 정확도와 오답 확인

<sub>실제 Windows 앱에서 합성 예제 이미지 3개를 GPU로 평가했습니다. 수치는 사용법 시연용입니다.</sub>

![CER와 Exact Match, 정답·예측·편집거리, 실행별 보고서를 보여주는 실제 평가 결과](docs/screenshots/results.png)

## 03. 이미지를 확대해 비교

![ROI 이미지를 확대해 정답과 예측을 비교하는 실제 앱 화면](docs/screenshots/zoom.png)

## 04. Dataset을 train / val / test로 나누기

<sub>캡처 묶음 선택 → 직접 배정 → 검사 → 확정 → 목록 저장. Windows 앱에서 예제 PNG로 확인한 화면입니다.</sub>

![캡처 이벤트별 배정과 확정 잠금, 용도별 개수 및 목록 저장을 보여주는 Dataset 화면](docs/screenshots/dataset.png)

## 05. 두 보고서의 성적과 샘플 비교

<sub>A 보고서 → B 보고서 → 비교. 고정 예제 보고서를 Windows 실제 앱에서 연 화면입니다.</sub>

![저장된 성적과 새로 틀림·새로 맞힘·둘 다 성공·둘 다 실패를 보여주는 결과 비교 화면](docs/screenshots/comparison.png)

## 06. 사전에서 빠진 문자 확인

<sub>학습 결과 폴더 → labels.json → 문자 검사. Windows 실제 앱에서 고정 예제의 문자 빈도·포함률을 확인한 화면입니다.</sub>

![고유 문자 수, 사전 포함률, 공백과 이모지를 포함한 문자별 등장 횟수와 누락 여부](docs/screenshots/charset.png)

## 07. 사진이 모델에 들어가는 과정 확인

<sub>평가 사진 → 모델 입력 확인 → 상세 정보. Windows 실제 모델에서 합성 예제의 전처리와 학습·평가 모드 크기를 확인했습니다.</sub>

![실제 전처리 단계와 패딩, 최종 입력의 shape·dtype·값 범위 및 학습·평가 모드 크기](docs/screenshots/model-input.png)
