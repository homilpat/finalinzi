# 구현 계획 - 스마트폰 센서와 WebView 연동

스마트폰 센서(가속도계, 자이로스코프)를 안드로이드 앱에 통합하고, `WebView`를 사용하여 기존 Flask 기반 웹 서비스와 브릿지를 구축합니다. 이를 통해 웹 서비스가 운동 중 사용자의 동작을 감지할 수 있도록 합니다.

## 사용자 검토 필요 사항

> [!IMPORTANT]
> 앱이 Flask 서비스를 로드하기 위해 **인터넷 권한(Internet Permission)**이 필요하며, **하드웨어 센서 접근** 권한이 필요합니다.
> 제안된 설계에 따라 동작 감지 로직이 정확하게 작동하려면 스마트폰을 **허리 뒤쪽**에 부착해야 합니다.

## 제안된 변경 사항

### 안드로이드 매니페스트 및 리소스

#### [MODIFY] [AndroidManifest.xml](file:///D:/newproject/app/src/main/AndroidManifest.xml)
- `android.permission.INTERNET` 권한을 추가합니다.
- 필요한 경우 가속도계 및 자이로스코프에 대한 하드웨어 기능 요구 사항을 추가합니다.

#### [MODIFY] [activity_main.xml](file:///D:/newproject/app/src/main/res/layout/activity_main.xml)
- 기존 `TextView`를 운동 서비스를 호스팅할 `WebView`로 교체합니다.

### 센서 로직 (핵심 감지)

#### [NEW] [SensorPreprocessor.kt](file:///D:/newproject/app/src/main/java/com/example/finalprojectapp/SensorPreprocessor.kt)
- 저역 통과 필터(LPF)를 사용하여 중력 성분과 선형 가속도를 분리하는 로직을 구현합니다.

#### [NEW] [Detectors.kt](file:///D:/newproject/app/src/main/java/com/example/finalprojectapp/Detectors.kt)
- 개별 동작 감지기 클래스를 포함합니다:
    - `StopDetector`: 사용자가 정지 상태인지 확인합니다.
    - `WeightShiftDetector`: 중력을 이용하여 측면 체중 이동을 확인합니다.
    - `StepDetector`: 충격 피크와 방향성 가속도를 이용하여 스텝(전진, 후진, 측면)을 감지합니다.
    - `KneeExtensionDetector`: 회전 또는 움직임 패턴을 기반으로 무릎 펴기(앉은 상태)를 감지합니다.
    - `AnyReactionDetector`: 유의미한 움직임이 발생했는지 감지합니다.

#### [NEW] [MotionClassifier.kt](file:///D:/newproject/app/src/main/java/com/example/finalprojectapp/MotionClassifier.kt)
- 감지기들을 관리하고 조율합니다.
- 원시 센서 데이터를 가져와 전처리하고, 기대되는 동작에 맞는 감지기를 쿼리합니다.
- 디바운싱(중복 방지) 처리를 담당합니다.

#### [NEW] [CalibrationProfile.kt](file:///D:/newproject/app/src/main/java/com/example/finalprojectapp/CalibrationProfile.kt)
- 사용자의 스마트폰 부착 방향에 따른 축 매핑 정보를 저장합니다.

### WebView 브릿지

#### [MODIFY] [MainActivity.kt](file:///D:/newproject/app/src/main/java/com/example/finalprojectapp/MainActivity.kt)
- JavaScript가 활성화된 `WebView`를 설정합니다.
- 다음 메서드를 포함하는 `WebAppInterface` (JavascriptInterface)를 구현합니다:
    - `startMeasurement(stage: String)`
    - `stopMeasurement()`
    - `setExpectedAction(action: String)`
- `onSensorChanged`의 센서 이벤트를 `MotionClassifier`로 전달합니다.
- 동작이 감지되면 WebView에서 `window.SensorBridge.onSensorEvent`를 호출합니다.

## 검증 계획

### 자동 테스트
- (시간이 허용되는 경우) 시뮬레이션된 센서 데이터를 사용하여 `MotionClassifier` 로직에 대한 단위 테스트를 수행합니다.

### 수동 검증
- 안드로이드 기기에 배포합니다.
- 앱을 열고 서비스를 로드합니다.
- 센서 등록/해제가 예상대로 작동하는지 확인합니다.
- 나열된 11가지 운동을 수행하고 올바른 액션이 WebView로 전송되는지 확인합니다.
- 정지 상태에서 "Stop" 감지를 확인합니다.
- 기울인 상태에서 "Weight Shift" 감지를 확인합니다.
- 움직이는 동안 "Steps" 감지를 확인합니다.
