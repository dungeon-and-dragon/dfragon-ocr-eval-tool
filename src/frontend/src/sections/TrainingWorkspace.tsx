import { useEffect, useState } from 'react'
import type { LibraryModel, TrainingOptions, TrainingSnapshot } from '../../../shared/training'
import type { Result } from '../../../shared/contracts'
import { PathField } from '../components/PathField'
import { SupplementSettings } from './SupplementSettings'
import { EvaluationResults } from '../components/EvaluationResults'

const initial: TrainingSnapshot = {
  status: 'idle',
  message: '',
  error: null,
  directory: null,
  counts: null,
  model: null,
  epoch: 0,
  epochs: 0,
  metrics: null,
  samples: [],
  publishedModelId: null
}

export function TrainingWorkspace(): React.JSX.Element {
  const [snapshot, setSnapshot] = useState(initial)
  const [models, setModels] = useState<LibraryModel[]>([])
  const [modelId, setModelId] = useState('')
  const [directory, setDirectory] = useState('')
  const [name, setName] = useState('')
  const [useSupplement, setUseSupplement] = useState(false)
  const [selectedSupplement, setSelectedSupplement] = useState<{
    directory: string
    id: string
  } | null>(null)
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [options, setOptions] = useState<TrainingOptions>({
    pythonExecutable: '',
    upstreamDirectory: '',
    epochs: 10,
    batchSize: 8,
    learningRate: 0.00001,
    nicknameDetachedGap: '4/17'
  })
  const active = [
    'downloading',
    'preparing',
    'training',
    'evaluating',
    'cancelling',
    'publishing'
  ].includes(snapshot.status)
  const busy = pending || active

  useEffect(() => {
    let disposed = false
    let received = false
    const unsubscribe = window.training.onSnapshot((next) => {
      received = true
      if (!disposed) {
        setSnapshot(next)
      }
    })
    void window.training
      .getSnapshot()
      .then((result) => {
        if (!disposed && !received && result.ok) {
          setSnapshot(result.value)
        }
      })
      .catch(() => {
        if (!disposed) {
          setError('학습 상태를 읽지 못했습니다.')
        }
      })
    void window.training
      .defaults()
      .then((result) => {
        if (!disposed && result.ok) {
          setOptions((current) => ({ ...current, ...result.value }))
        }
      })
      .catch(() => {
        if (!disposed) {
          setError('학습 환경 설정을 읽지 못했습니다.')
        }
      })
    return () => {
      disposed = true
      unsubscribe()
    }
  }, [])

  async function command<T>(
    action: () => Promise<Result<T>>,
    success?: (value: T) => void
  ): Promise<void> {
    if (pending) {
      return
    }
    setPending(true)
    setError(null)
    try {
      const result = await action()
      if (result.ok) {
        success?.(result.value)
      } else {
        setError(result.error)
      }
    } catch {
      setError('요청을 처리하지 못했습니다. 현재 작업 상태를 확인해 주세요.')
    } finally {
      setPending(false)
    }
  }

  async function refreshModels(): Promise<void> {
    const result = await window.training.models()
    if (!result.ok) {
      setError(result.error)
      return
    }
    setModels(result.value)
    setModelId((current) =>
      result.value.some((model) => model.id === current) ? current : (result.value[0]?.id ?? '')
    )
  }

  function choose(kind: 'python' | 'source' | 'output'): void {
    void command(
      () => window.evaluation.choosePath(kind),
      (value) => {
        if (value === null) {
          return
        }
        if (kind === 'output') {
          setDirectory(value)
        } else {
          setOptions((current) => ({
            ...current,
            [kind === 'python' ? 'pythonExecutable' : 'upstreamDirectory']: value
          }))
        }
      }
    )
  }

  const failure = error ?? snapshot.error
  const complete = snapshot.status === 'completed' || snapshot.status === 'publishing'
  return (
    <>
      {failure !== null && (
        <div className="error-banner" role="alert">
          {failure}
        </div>
      )}
      <div className="workspace">
        <section className="setup-panel panel" aria-label="자료실 학습 설정">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">OCR LIBRARY → LOCAL GPU</span>
              <h2>모델 학습</h2>
            </div>
          </div>
          <p>미니PC의 모델과 데이터를 가져와 이 PC에서 학습합니다.</p>
          <div className="training-actions">
            <button
              className="button button-secondary"
              disabled={busy}
              onClick={() =>
                void command(async () => {
                  const result = await window.training.login()
                  if (result.ok) {
                    await refreshModels()
                  }
                  return result
                })
              }
            >
              자료실 로그인
            </button>
            <button
              className="button button-secondary"
              disabled={busy}
              onClick={() =>
                void command(async () => {
                  await refreshModels()
                  return { ok: true, value: null }
                })
              }
            >
              모델 목록 새로고침
            </button>
            <button
              className="button button-secondary"
              disabled={busy}
              onClick={() =>
                void command(
                  () => window.training.logout(),
                  () => {
                    setModels([])
                    setModelId('')
                  }
                )
              }
            >
              로그아웃
            </button>
          </div>
          <div className="field">
            <label htmlFor="training-model">시작 모델</label>
            <select
              id="training-model"
              value={modelId}
              disabled={busy}
              onChange={(event) => setModelId(event.target.value)}
            >
              <option value="">자료실에서 모델을 선택하세요</option>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.name} · {model.kind === 'pretrained' ? '기본' : '파인튜닝'}
                </option>
              ))}
            </select>
          </div>
          <p className="field-help">
            등록된 모델이 없다면 자료실 웹의 ‘학습 모델’에서 기본 한국어 모델을 추가하세요.
          </p>
          <button
            className="button button-secondary"
            disabled={busy}
            onClick={() =>
              void command(async () => {
                const selected = await window.evaluation.choosePath('output')
                if (!selected.ok || selected.value === null) {
                  return selected
                }
                return window.training.open(selected.value)
              })
            }
          >
            저장한 실험·결과 열기
          </button>
          <p className="field-help">
            실험 폴더를 열어 다시 학습하거나, 그 안의 runs/결과 폴더를 열어 성적을 확인하고 등록할
            수 있습니다.
          </p>
          <PathField
            id="training-output"
            label="실험 저장 폴더"
            value={directory}
            placeholder="모델·데이터·학습 결과를 보관할 폴더"
            disabled={busy}
            onChoose={() => choose('output')}
          />
          <button
            className="button button-primary"
            disabled={busy || !modelId || !directory}
            onClick={() => void command(() => window.training.download(modelId, directory))}
          >
            모델·데이터 가져오기
          </button>
          {snapshot.counts !== null && (
            <p role="status">
              train {snapshot.counts.train} · val {snapshot.counts.val} · test{' '}
              {snapshot.counts.test} · 사용 안 함 {snapshot.counts.skipped}
            </p>
          )}
          <div className="training-options">
            <label className="field">
              학습 횟수 (epoch)
              <input
                type="number"
                min="1"
                max="1000"
                value={options.epochs}
                disabled={busy}
                onChange={(event) => setOptions({ ...options, epochs: Number(event.target.value) })}
              />
            </label>
            <label className="field">
              배치 크기
              <input
                type="number"
                min="1"
                max="128"
                value={options.batchSize}
                disabled={busy}
                onChange={(event) =>
                  setOptions({ ...options, batchSize: Number(event.target.value) })
                }
              />
            </label>
            <label className="field">
              학습률
              <input
                type="number"
                min="0.00000001"
                max="0.1"
                step="0.00001"
                value={options.learningRate}
                disabled={busy}
                onChange={(event) =>
                  setOptions({ ...options, learningRate: Number(event.target.value) })
                }
              />
            </label>
            <label className="field">
              닉네임 이진화의 떨어진 픽셀 간격
              <select
                value={options.nicknameDetachedGap}
                disabled={busy}
                onChange={(event) =>
                  setOptions({
                    ...options,
                    nicknameDetachedGap: event.target
                      .value as TrainingOptions['nicknameDetachedGap']
                  })
                }
              >
                <option value="4/17">높이의 4/17 (던삐삐 main)</option>
                <option value="6/17">높이의 6/17 (던삐삐 #1152)</option>
              </select>
            </label>
          </div>
          <details className="training-runtime">
            <summary>학습 실행 환경</summary>
            <PathField
              id="training-python"
              label="학습 Python"
              value={options.pythonExecutable}
              placeholder="Paddle GPU 환경의 Python"
              disabled={busy}
              onChoose={() => choose('python')}
            />
            <PathField
              id="training-source"
              label="PaddleOCR 소스"
              value={options.upstreamDirectory}
              placeholder="지원 버전의 PaddleOCR 폴더"
              disabled={busy}
              onChoose={() => choose('source')}
            />
          </details>
          <button
            className="button button-primary"
            disabled={
              busy ||
              snapshot.directory === null ||
              !snapshot.model ||
              !options.pythonExecutable ||
              !options.upstreamDirectory ||
              (useSupplement &&
                (!selectedSupplement || selectedSupplement.directory !== snapshot.directory))
            }
            onClick={() =>
              void command(() =>
                window.training.start({
                  ...options,
                  ...(useSupplement && selectedSupplement
                    ? { supplementId: selectedSupplement.id }
                    : {})
                })
              )
            }
          >
            학습 후 test 평가
          </button>
          <p className="field-help">
            train으로 학습하고 val의 CER로 모델을 선택합니다. 선택한 모델만 test로 최종 평가합니다.
          </p>
          {snapshot.model && (
            <p className="field-help">학습에 사용할 모델: {snapshot.model.name}</p>
          )}
        </section>
        <div className="results-column">
          {snapshot.directory && (
            <SupplementSettings
              key={snapshot.directory}
              directory={snapshot.directory}
              busy={busy}
              enabled={useSupplement}
              setEnabled={setUseSupplement}
              additionalCharacters={options.additionalCharacters ?? ''}
              setAdditionalCharacters={(value) => {
                setOptions((current) => ({ ...current, additionalCharacters: value }))
                setSelectedSupplement(null)
              }}
              onSelect={(id) =>
                setSelectedSupplement(
                  id && snapshot.directory ? { id, directory: snapshot.directory } : null
                )
              }
              saved={snapshot.supplement}
            />
          )}
          {!!snapshot.addedCharacters?.length && (
            <p>저장된 모델의 추가 문자: {snapshot.addedCharacters.join('')}</p>
          )}
          <section className="panel training-progress" aria-label="학습 진행 상태">
            <h3>{snapshot.model?.name ?? '학습 준비'}</h3>
            <p role="status">
              {snapshot.message || '자료실에 로그인하고 모델·데이터를 가져오세요.'}
            </p>
            {snapshot.epochs > 0 && (
              <progress
                value={snapshot.epoch}
                max={snapshot.epochs}
                aria-label="학습 epoch 진행률"
              />
            )}
            {active && (
              <button
                className="button button-secondary"
                disabled={snapshot.status === 'cancelling'}
                onClick={() =>
                  void window.training.cancel().then((result) => {
                    if (!result.ok) {
                      setError(result.error)
                    }
                  })
                }
              >
                작업 취소
              </button>
            )}
          </section>
          <section className="metrics-grid" aria-label="학습 모델의 test 성적">
            <div className="metric-card">
              <span>CER</span>
              <strong>
                {complete && snapshot.metrics ? `${(snapshot.metrics.cer * 100).toFixed(2)}%` : '—'}
              </strong>
              <small>문자 오류율 · 낮을수록 좋음</small>
            </div>
            <div className="metric-card">
              <span>Exact Match</span>
              <strong>
                {complete && snapshot.metrics
                  ? `${(snapshot.metrics.exactMatch * 100).toFixed(2)}%`
                  : '—'}
              </strong>
              <small>완전히 맞힌 샘플 비율</small>
            </div>
            <div className="metric-card">
              <span>test 샘플</span>
              <strong>{complete && snapshot.metrics ? snapshot.metrics.sampleCount : '—'}</strong>
              <small>최종 평가 데이터</small>
            </div>
          </section>
          <EvaluationResults
            samples={snapshot.samples}
            status={
              complete
                ? 'completed'
                : active
                  ? 'running'
                  : snapshot.status === 'failed'
                    ? 'failed'
                    : snapshot.status === 'cancelled'
                      ? 'cancelled'
                      : 'idle'
            }
            readImage={window.training.readImage}
          />
          {complete && (
            <section className="panel training-publish" aria-label="학습 결과 등록">
              <h3>미니PC에 보관</h3>
              <p>새 모델과 평가 성적을 자료실에 등록합니다. 기존 모델은 유지됩니다.</p>
              <label className="field">
                새 모델 이름
                <input
                  value={name}
                  maxLength={100}
                  disabled={busy || snapshot.publishedModelId !== null}
                  onChange={(event) => setName(event.target.value)}
                  placeholder="예: 닉네임 모델 · 1차 학습"
                />
              </label>
              <button
                className="button button-primary"
                disabled={busy || !name.trim() || snapshot.publishedModelId !== null}
                onClick={() => void command(() => window.training.publish(name))}
              >
                {snapshot.publishedModelId ? '자료실 등록 완료' : '미니PC에 올리기'}
              </button>
            </section>
          )}
        </div>
      </div>
    </>
  )
}
