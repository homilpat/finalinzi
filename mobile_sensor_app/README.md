# Finalinzi Mobile Sensor

React Native / Expo sensor collector for iPhone and Android.

This app only collects sensor data and exports the same 14-column CSV shape used by the existing Android APK samples.

## CSV Format

```csv
Timestamp_ns,Acc_X,Acc_Y,Acc_Z,Gyro_Raw_X,Gyro_Raw_Y,Gyro_Raw_Z,Gyro_Clean_X,Gyro_Clean_Y,Gyro_Clean_Z,Acc_Vertical_g,Acc_ML_g,Acc_AP_g,Gyro_Roll_deg_s
```

Header comments also include:

- `Gyro_Zero_Bias_rad_s`
- `Gravity_Mean_m_s2`
- `Basis_Vertical_Unit`
- `Basis_ML_Unit`
- `Basis_AP_Unit`

## Measurement Types

- `gait`: 보행 측정
- `knee_raise`: 제자리 무릎 들어올리기
- `jump_stop`: 제자리뛰기 후 급정지
- `side_walk`: 사이드 걷기
- `seated_knee_extension`: 의자에 앉은 상태로 양쪽 무릎 펴기

All modes use the same CSV columns. The selected mode is also written in the CSV comments:

```text
# Session_Type: knee_raise
# Session_Label: 제자리 무릎 들어올리기
```

For Flask exercise analysis, send the CSV plus the matching `exercise_type` field:

```http
POST /exercise/sensor/analyze
Content-Type: multipart/form-data

file: sensor.csv
exercise_type: knee_raise
```

## Measurement Protocol

Gait mode:

```text
7초 준비
-> 3초 정지 gravity / gyro bias calibration
-> 20초 보행 측정
-> CSV 저장/공유
```

Exercise modes:

```text
3초 준비
-> 사용자가 멈출 때까지 동작 측정
-> 측정 종료/CSV 저장 버튼
-> CSV 공유/저장
```

## Run

```bash
cd mobile_sensor_app
npm install
npx expo install expo-sensors expo-file-system expo-sharing
npm run start
```

Then scan the Expo QR code on iPhone or Android.

## Notes

- Expo `Accelerometer` returns acceleration in `g`; this app converts raw `Acc_X/Y/Z` to `m/s^2` to match Android sensor CSV samples.
- Expo `Gyroscope` returns `rad/s`, matching the existing APK raw gyro columns.
- Anatomical V / ML / AP columns are computed from the 3-second still gravity vector.
- Server inference is not included here. Share/upload the CSV to Flask `/gait/upload-csv`.
