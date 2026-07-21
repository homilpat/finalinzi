import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Alert,
  Platform,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { Accelerometer, Gyroscope } from 'expo-sensors';
import { Directory, EncodingType, File, Paths } from 'expo-file-system';
import * as Sharing from 'expo-sharing';

const G = 9.80665;
const SAMPLE_INTERVAL_MS = 10;
const UI_SAMPLE_INTERVAL_MS = 250;

const SERVER_URL = 'https://moca-demo.onrender.com';

const SESSION_TYPES = [
  {
    key: 'gait',
    label: '보행 측정',
    instruction: '평소처럼 20초 동안 걸어주세요.',
    filenamePrefix: 'gait',
    prepSec: 7,
    calibrationSec: 3,
    measureSec: 20,
  },
  {
    key: 'knee_raise',
    label: '제자리 무릎 들어올리기',
    instruction: '제자리에서 무릎을 번갈아 들어올려 주세요.',
    filenamePrefix: 'knee_raise',
    prepSec: 3,
    calibrationSec: 0,
    measureSec: null,
  },
  {
    key: 'jump_stop',
    label: '제자리뛰기 후 급정지',
    instruction: '제자리뛰기를 하다가 마지막에 멈춰 주세요.',
    filenamePrefix: 'jump_stop',
    prepSec: 3,
    calibrationSec: 0,
    measureSec: null,
  },
  {
    key: 'side_walk',
    label: '사이드 걷기',
    instruction: '오른쪽 또는 왼쪽으로 사이드 스텝을 해주세요.',
    filenamePrefix: 'side_walk',
    prepSec: 3,
    calibrationSec: 0,
    measureSec: null,
  },
  {
    key: 'seated_knee_extension',
    label: '앉아서 양쪽 무릎 펴기',
    instruction: '의자에 앉아 양쪽 무릎을 펴는 동작을 반복해 주세요.',
    filenamePrefix: 'seated_knee_extension',
    prepSec: 3,
    calibrationSec: 0,
    measureSec: null,
  },
];

const PHASE = {
  idle: 'idle',
  prep: 'prep',
  calibration: 'calibration',
  walking: 'walking',
  done: 'done',
};

function nowNs() {
  return Math.round(Date.now() * 1e6);
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function norm(v) {
  return Math.sqrt(dot(v, v));
}

function normalize(v, fallback = [0, 1, 0]) {
  const n = norm(v);
  if (!Number.isFinite(n) || n < 1e-9) return fallback;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

function sub(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function mul(v, scalar) {
  return [v[0] * scalar, v[1] * scalar, v[2] * scalar];
}

function meanVec(rows, selector) {
  if (!rows.length) return [0, G, 0];
  const sum = rows.reduce(
    (acc, row) => {
      const v = selector(row);
      acc[0] += v[0];
      acc[1] += v[1];
      acc[2] += v[2];
      return acc;
    },
    [0, 0, 0],
  );
  return [sum[0] / rows.length, sum[1] / rows.length, sum[2] / rows.length];
}

function buildBasis(gravityMean) {
  const vertical = normalize(gravityMean, [0, 1, 0]);
  const refs = [
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
  ];
  let ap = [0, 0, 1];
  for (const ref of refs) {
    const projected = sub(ref, mul(vertical, dot(ref, vertical)));
    if (norm(projected) > 0.05) {
      ap = normalize(projected, [0, 0, 1]);
      break;
    }
  }
  const ml = normalize(cross(vertical, ap), [1, 0, 0]);
  return { vertical, ml, ap };
}

function toCsvNumber(value) {
  if (!Number.isFinite(value)) return '';
  return Number(value).toPrecision(9);
}

function formatVec(v) {
  return `(${toCsvNumber(v[0])}, ${toCsvNumber(v[1])}, ${toCsvNumber(v[2])})`;
}

function buildCsv(rows, calibrationRows, basis, gyroBias, sessionType) {
  const calibrationSource = calibrationRows.length ? calibrationRows : rows.slice(0, Math.min(rows.length, 50));
  const gravityMean = meanVec(calibrationSource, (row) => row.acc);
  const header = [
    '# [Finalinzi RN] 14-Column Anatomical Sensor Dataset',
    `# Session_Type: ${sessionType.key}`,
    `# Session_Label: ${sessionType.label}`,
    `# Gyro_Zero_Bias_rad_s: Gx=${toCsvNumber(gyroBias[0])},Gy=${toCsvNumber(gyroBias[1])},Gz=${toCsvNumber(gyroBias[2])}`,
    `# Gravity_Mean_m_s2: Ax=${toCsvNumber(gravityMean[0])},Ay=${toCsvNumber(gravityMean[1])},Az=${toCsvNumber(gravityMean[2])}`,
    `# Basis_Vertical_Unit: ${formatVec(basis.vertical)}`,
    `# Basis_ML_Unit: ${formatVec(basis.ml)}`,
    `# Basis_AP_Unit: ${formatVec(basis.ap)}`,
    'Timestamp_ns,Acc_X,Acc_Y,Acc_Z,Gyro_Raw_X,Gyro_Raw_Y,Gyro_Raw_Z,Gyro_Clean_X,Gyro_Clean_Y,Gyro_Clean_Z,Acc_Vertical_g,Acc_ML_g,Acc_AP_g,Gyro_Roll_deg_s',
  ];

  const body = rows.map((row) => {
    const cleanGyro = [
      row.gyro[0] - gyroBias[0],
      row.gyro[1] - gyroBias[1],
      row.gyro[2] - gyroBias[2],
    ];
    const accVerticalG = dot(row.acc, basis.vertical) / G;
    const accMlG = dot(row.acc, basis.ml) / G;
    const accApG = dot(row.acc, basis.ap) / G;
    const gyroRollDegS = dot(cleanGyro, basis.ap) * (180 / Math.PI);
    return [
      row.timestampNs,
      row.acc[0],
      row.acc[1],
      row.acc[2],
      row.gyro[0],
      row.gyro[1],
      row.gyro[2],
      cleanGyro[0],
      cleanGyro[1],
      cleanGyro[2],
      accVerticalG,
      accMlG,
      accApG,
      gyroRollDegS,
    ]
      .map(toCsvNumber)
      .join(',');
  });
  return `${header.concat(body).join('\n')}\n`;
}

export default function App() {
  const [phase, setPhase] = useState(PHASE.idle);
  const [selectedTypeKey, setSelectedTypeKey] = useState('gait');
  const [secondsLeft, setSecondsLeft] = useState(0);
  const [sampleCount, setSampleCount] = useState(0);
  const [csvPath, setCsvPath] = useState('');
  const [csvFilename, setCsvFilename] = useState('');
  const [summary, setSummary] = useState('허리에 폰을 고정하고 시작 버튼을 눌러주세요.');
  const [uploading, setUploading] = useState(false);
  const [gaitResult, setGaitResult] = useState(null);
  const latestAcc = useRef([0, G, 0]);
  const latestGyro = useRef([0, 0, 0]);
  const calibrationRows = useRef([]);
  const measurementRows = useRef([]);
  const timerRef = useRef(null);
  const phaseRef = useRef(PHASE.idle);
  const lastSampleUiMs = useRef(0);

  const selectedType = useMemo(
    () => SESSION_TYPES.find((item) => item.key === selectedTypeKey) || SESSION_TYPES[0],
    [selectedTypeKey],
  );
  const isManualExercise = selectedType.measureSec == null;
  const totalSec = selectedType.prepSec + selectedType.calibrationSec + (selectedType.measureSec ?? 0);

  const phaseLabel = useMemo(() => {
    if (phase === PHASE.prep) return '준비';
    if (phase === PHASE.calibration) return '3초 정지 보정';
    if (phase === PHASE.walking) return isManualExercise ? '동작 측정 중' : '20초 보행 측정';
    if (phase === PHASE.done) return 'CSV 생성 완료';
    return '대기';
  }, [phase, isManualExercise]);

  useEffect(() => {
    Accelerometer.setUpdateInterval(SAMPLE_INTERVAL_MS);
    Gyroscope.setUpdateInterval(SAMPLE_INTERVAL_MS);

    const accSub = Accelerometer.addListener((data) => {
      latestAcc.current = [data.x * G, data.y * G, data.z * G];
      const currentPhase = phaseRef.current;
      if (currentPhase !== PHASE.calibration && currentPhase !== PHASE.walking) return;

      const row = {
        timestampNs: nowNs(),
        acc: latestAcc.current,
        gyro: latestGyro.current,
      };

      if (currentPhase === PHASE.calibration) {
        calibrationRows.current.push(row);
      } else {
        measurementRows.current.push(row);
        const uiNow = Date.now();
        if (uiNow - lastSampleUiMs.current >= UI_SAMPLE_INTERVAL_MS) {
          lastSampleUiMs.current = uiNow;
          setSampleCount(measurementRows.current.length);
        }
      }
    });

    const gyroSub = Gyroscope.addListener((data) => {
      latestGyro.current = [data.x, data.y, data.z];
    });

    return () => {
      accSub.remove();
      gyroSub.remove();
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  function transition(nextPhase, nextSeconds) {
    phaseRef.current = nextPhase;
    setPhase(nextPhase);
    setSecondsLeft(nextSeconds);
  }

  async function startMeasurement() {
    try {
      const accAvailable = await Accelerometer.isAvailableAsync();
      const gyroAvailable = await Gyroscope.isAvailableAsync();
      if (!accAvailable || !gyroAvailable) {
        Alert.alert('센서 확인', '가속도계 또는 자이로스코프를 사용할 수 없습니다.');
        return;
      }

      const accPermission = await Accelerometer.requestPermissionsAsync();
      const gyroPermission = await Gyroscope.requestPermissionsAsync();
      if (!accPermission.granted || !gyroPermission.granted) {
        Alert.alert('센서 권한 필요', '측정을 위해 모션 센서 권한을 허용해 주세요.');
        return;
      }

      calibrationRows.current = [];
      measurementRows.current = [];
      lastSampleUiMs.current = 0;
      setCsvPath('');
      setCsvFilename('');
      setSampleCount(0);
      setSummary(
        isManualExercise
          ? '3초 뒤 동작 측정을 시작합니다.'
          : '7초 동안 자세를 잡고, 보정 단계에서는 3초간 정지해 주세요.',
      );
      transition(PHASE.prep, totalSec);

      const startMs = Date.now();
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = setInterval(async () => {
        const elapsed = (Date.now() - startMs) / 1000;
        const remaining = isManualExercise
          ? Math.max(0, Math.ceil(selectedType.prepSec - elapsed))
          : Math.max(0, Math.ceil(totalSec - elapsed));
        setSecondsLeft(remaining);

        if (selectedType.prepSec > 0 && elapsed < selectedType.prepSec) {
          if (phaseRef.current !== PHASE.prep) transition(PHASE.prep, remaining);
        } else if (
          selectedType.calibrationSec > 0 &&
          elapsed < selectedType.prepSec + selectedType.calibrationSec
        ) {
          if (phaseRef.current !== PHASE.calibration) {
            setSummary('움직이지 말고 3초간 정지해 주세요.');
            transition(PHASE.calibration, remaining);
          }
        } else if (isManualExercise) {
          if (phaseRef.current !== PHASE.walking) {
            setSummary(`${selectedType.instruction} 끝나면 아래 버튼을 눌러 CSV를 저장하세요.`);
            transition(PHASE.walking, 0);
          }
        } else if (elapsed < totalSec) {
          if (phaseRef.current !== PHASE.walking) {
            setSummary(selectedType.instruction);
            transition(PHASE.walking, remaining);
          }
        } else {
          clearInterval(timerRef.current);
          timerRef.current = null;
          await finishMeasurement();
        }
      }, 200);
    } catch (error) {
      transition(PHASE.idle, 0);
      setSummary('센서 시작 중 오류가 났습니다. 권한을 확인한 뒤 다시 시도해 주세요.');
      Alert.alert('센서 오류', error?.message || '알 수 없는 오류');
    }
  }

  async function finishMeasurement() {
    const calibration = calibrationRows.current;
    const measurement = measurementRows.current;
    setSampleCount(measurement.length);

    const calibrationOk = selectedType.calibrationSec > 0 ? calibration.length >= 10 : true;
    const minSamples = isManualExercise ? 1 : 100;
    if (!calibrationOk || measurement.length < minSamples) {
      transition(PHASE.idle, 0);
      setSummary(`샘플이 부족합니다. 현재 ${measurement.length}개입니다. 다시 측정해 주세요.`);
      return '';
    }

    const calibrationSource = calibration.length ? calibration : measurement.slice(0, Math.min(measurement.length, 50));
    const gravityMean = meanVec(calibrationSource, (row) => row.acc);
    const gyroBias = meanVec(calibrationSource, (row) => row.gyro);
    const basis = buildBasis(gravityMean);
    const csv = buildCsv(measurement, calibration, basis, gyroBias, selectedType);
    const filename = `finalinzi_${selectedType.filenamePrefix}_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
    const file = new File(Paths.document, filename);
    file.create({ overwrite: true });
    file.write(csv, { encoding: EncodingType.UTF8 });
    setCsvPath(file.uri);
    setCsvFilename(filename);
    transition(PHASE.done, 0);
    setSummary(`CSV 생성 완료: ${selectedType.label}, ${measurement.length}개 샘플`);
    return file.uri;
  }

  async function stopManualMeasurement() {
    if (!isManualExercise || phaseRef.current !== PHASE.walking) return '';
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    return finishMeasurement();
  }

  async function handlePrimaryPress() {
    if (phase === PHASE.walking && isManualExercise) {
      const path = await stopManualMeasurement();
      if (path) await saveCsvToPhone(path);
      return;
    }
    await startMeasurement();
  }

  async function saveCsvToPhone(path) {
    if (!path) return;

    try {
      const filename = csvFilename || path.split('/').pop() || 'finalinzi_sensor.csv';

      if (Platform.OS === 'android') {
        const sourceFile = new File(path);
        const targetDir = await Directory.pickDirectoryAsync();
        const targetFile = targetDir.createFile(filename, 'text/csv');
        targetFile.write(await sourceFile.text(), { encoding: EncodingType.UTF8 });
        setSummary(`휴대폰 선택 폴더에 저장 완료: ${filename}`);
        Alert.alert('CSV 저장 완료', `${filename} 파일을 선택한 폴더에 저장했습니다.`);
        return;
      }

      const canShare = await Sharing.isAvailableAsync();
      if (!canShare) {
        Alert.alert('CSV 생성됨', `파일 위치:\n${path}`);
        return;
      }
      await Sharing.shareAsync(path, {
        mimeType: 'text/csv',
        dialogTitle: `${selectedType.label} CSV 저장`,
        UTI: 'public.comma-separated-values-text',
      });
    } catch (error) {
      Alert.alert('CSV 저장 오류', `${error?.message || '알 수 없는 오류'}\n\n앱 내부 파일:\n${path}`);
    }
  }

  async function handleSavePress() {
    let path = csvPath;
    if (!path && phaseRef.current === PHASE.walking && isManualExercise) {
      if (timerRef.current) clearInterval(timerRef.current);
      timerRef.current = null;
      path = await finishMeasurement();
    }

    if (!path) {
      Alert.alert('CSV 없음', '측정이 끝난 뒤 CSV를 저장할 수 있습니다.');
      return;
    }
    await saveCsvToPhone(path);
  }

  async function uploadToServer() {
    if (!csvPath) return;
    setUploading(true);
    setGaitResult(null);
    try {
      const filename = csvFilename || csvPath.split('/').pop() || 'gait.csv';
      const formData = new FormData();
      formData.append('file', { uri: csvPath, name: filename, type: 'text/csv' });
      const res = await fetch(`${SERVER_URL}/gait/upload-csv`, {
        method: 'POST',
        body: formData,
      });
      const json = await res.json();
      if (!res.ok || !json.ok) {
        Alert.alert('서버 오류', json.error || '분석 실패');
        return;
      }
      setGaitResult(json);
    } catch (e) {
      Alert.alert('전송 실패', e?.message || '네트워크 오류');
    } finally {
      setUploading(false);
    }
  }

  function reset() {
    if (timerRef.current) clearInterval(timerRef.current);
    timerRef.current = null;
    calibrationRows.current = [];
    measurementRows.current = [];
    lastSampleUiMs.current = 0;
    setCsvPath('');
    setCsvFilename('');
    setSampleCount(0);
    setGaitResult(null);
    transition(PHASE.idle, 0);
    setSummary('허리에 폰을 고정하고 시작 버튼을 눌러주세요.');
  }

  const primaryCanStop = phase === PHASE.walking && isManualExercise;
  const primaryDisabled = phase !== PHASE.idle && phase !== PHASE.done && !primaryCanStop;
  const primaryLabel = primaryCanStop ? '측정 종료/CSV 저장' : '측정 시작';

  return (
    <SafeAreaView style={styles.safe}>
      <ScrollView contentContainerStyle={styles.container}>
        <Text style={styles.title}>Finalinzi Sensor</Text>
        <Text style={styles.subtitle}>기존 APK와 동일한 14컬럼 CSV 수집</Text>

        <View style={styles.selector}>
          {SESSION_TYPES.map((item) => (
            <TouchableOpacity
              key={item.key}
              style={[
                styles.choice,
                selectedTypeKey === item.key && styles.choiceActive,
                phase !== PHASE.idle && phase !== PHASE.done && styles.choiceDisabled,
              ]}
              onPress={() => setSelectedTypeKey(item.key)}
              disabled={phase !== PHASE.idle && phase !== PHASE.done}
            >
              <Text style={[styles.choiceText, selectedTypeKey === item.key && styles.choiceTextActive]}>
                {item.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.panel}>
          <Text style={styles.label}>상태</Text>
          <Text style={styles.phase}>{phaseLabel}</Text>
          <Text style={styles.timer}>{secondsLeft > 0 ? `${secondsLeft}초` : '-'}</Text>
          <Text style={styles.summary}>{summary}</Text>
          <Text style={styles.samples}>측정 샘플: {sampleCount}</Text>
        </View>

        <TouchableOpacity
          style={[styles.button, primaryDisabled && styles.buttonDisabled]}
          onPress={handlePrimaryPress}
          disabled={primaryDisabled}
        >
          <Text style={styles.buttonText}>{primaryLabel}</Text>
        </TouchableOpacity>

        <TouchableOpacity style={[styles.button, styles.secondary, !csvPath && styles.buttonMuted]} onPress={handleSavePress}>
          <Text style={styles.secondaryText}>CSV 폰에 저장/공유</Text>
        </TouchableOpacity>

        {selectedTypeKey === 'gait' && (
          <TouchableOpacity
            style={[styles.button, styles.uploadBtn, (!csvPath || uploading) && styles.buttonDisabled]}
            onPress={uploadToServer}
            disabled={!csvPath || uploading}
          >
            <Text style={styles.buttonText}>{uploading ? '분석 중...' : '서버에 보행 분석 전송'}</Text>
          </TouchableOpacity>
        )}

        {gaitResult && (
          <View style={[styles.resultCard, gaitResult.prediction === 0 ? styles.resultNormal : styles.resultImpaired]}>
            <Text style={styles.resultLabel}>
              {gaitResult.prediction === 0 ? '정상' : '운동기능저하 의심'}
            </Text>
            <Text style={styles.resultProb}>
              확률 {(gaitResult.probability * 100).toFixed(1)}%
            </Text>
            <Text style={styles.resultSub}>
              임계값 {gaitResult.threshold}  |  {gaitResult.model_mode ?? ''}
            </Text>
          </View>
        )}

        <TouchableOpacity style={[styles.button, styles.ghost]} onPress={reset}>
          <Text style={styles.ghostText}>초기화</Text>
        </TouchableOpacity>

        <View style={styles.schema}>
          <Text style={styles.schemaTitle}>CSV columns</Text>
          <Text style={styles.schemaText}>
            Timestamp_ns, Acc_X, Acc_Y, Acc_Z, Gyro_Raw_X, Gyro_Raw_Y, Gyro_Raw_Z,
            Gyro_Clean_X, Gyro_Clean_Y, Gyro_Clean_Z, Acc_Vertical_g, Acc_ML_g,
            Acc_AP_g, Gyro_Roll_deg_s
          </Text>
          <Text style={styles.schemaNote}>
            이동 분석 서버 전송 시 exercise_type은{' '}
            {SESSION_TYPES.filter((item) => item.key !== 'gait')
              .map((item) => item.key)
              .join(', ')}{' '}
            중 하나를 같이 보내면 됩니다.
          </Text>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: '#eef4fb',
  },
  container: {
    padding: 24,
    gap: 16,
  },
  title: {
    fontSize: 32,
    fontWeight: '800',
    color: '#075ca8',
  },
  subtitle: {
    fontSize: 16,
    color: '#425466',
  },
  selector: {
    gap: 8,
  },
  choice: {
    minHeight: 44,
    borderRadius: 8,
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#cbd5e1',
    justifyContent: 'center',
  },
  choiceActive: {
    backgroundColor: '#1477c9',
    borderColor: '#1477c9',
  },
  choiceDisabled: {
    opacity: 0.55,
  },
  choiceText: {
    color: '#243b53',
    fontSize: 15,
    fontWeight: '700',
  },
  choiceTextActive: {
    color: '#ffffff',
  },
  panel: {
    backgroundColor: '#ffffff',
    borderRadius: 8,
    padding: 20,
    gap: 8,
  },
  label: {
    fontSize: 14,
    color: '#6b7a90',
    fontWeight: '700',
  },
  phase: {
    fontSize: 24,
    fontWeight: '800',
    color: '#102a43',
  },
  timer: {
    fontSize: 48,
    fontWeight: '900',
    color: '#1477c9',
  },
  summary: {
    fontSize: 16,
    color: '#243b53',
    lineHeight: 22,
  },
  samples: {
    fontSize: 16,
    color: '#1477c9',
    fontWeight: '700',
  },
  button: {
    minHeight: 56,
    borderRadius: 8,
    paddingHorizontal: 16,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1477c9',
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  buttonMuted: {
    opacity: 0.7,
  },
  buttonText: {
    color: '#ffffff',
    fontSize: 18,
    fontWeight: '800',
  },
  secondary: {
    backgroundColor: '#ffffff',
    borderWidth: 1,
    borderColor: '#1477c9',
  },
  secondaryText: {
    color: '#1477c9',
    fontSize: 18,
    fontWeight: '800',
    textAlign: 'center',
  },
  ghost: {
    backgroundColor: 'transparent',
  },
  ghostText: {
    color: '#425466',
    fontSize: 16,
    fontWeight: '700',
  },
  uploadBtn: {
    backgroundColor: '#0d9488',
  },
  resultCard: {
    borderRadius: 12,
    padding: 20,
    alignItems: 'center',
    gap: 6,
  },
  resultNormal: {
    backgroundColor: '#d1fae5',
  },
  resultImpaired: {
    backgroundColor: '#fee2e2',
  },
  resultLabel: {
    fontSize: 28,
    fontWeight: '900',
    color: '#0f172a',
  },
  resultProb: {
    fontSize: 20,
    fontWeight: '700',
    color: '#0f172a',
  },
  resultSub: {
    fontSize: 13,
    color: '#475569',
  },
  schema: {
    backgroundColor: '#dbeafe',
    borderRadius: 8,
    padding: 16,
  },
  schemaTitle: {
    fontSize: 15,
    fontWeight: '800',
    color: '#0f172a',
    marginBottom: 8,
  },
  schemaText: {
    fontSize: 13,
    color: '#334155',
    lineHeight: 19,
  },
  schemaNote: {
    marginTop: 10,
    fontSize: 13,
    color: '#475569',
    lineHeight: 19,
  },
});
