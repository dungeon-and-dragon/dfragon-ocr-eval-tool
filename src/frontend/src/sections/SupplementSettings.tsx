import {
  CHARACTER_GROUPS,
  CHARACTER_GROUP_LABELS,
  type SupplementPreview
} from '../../../shared/supplement'
import { useSupplementSettings } from '../hooks/useSupplementSettings'
import { SupplementDistribution } from '../components/SupplementDistribution'
import { SupplementAppearance } from '../components/SupplementAppearance'

export function SupplementSettings({
  directory,
  busy,
  enabled,
  setEnabled,
  additionalCharacters,
  setAdditionalCharacters,
  onSelect,
  saved
}: {
  directory: string
  busy: boolean
  enabled: boolean
  setEnabled: (enabled: boolean) => void
  additionalCharacters: string
  setAdditionalCharacters: (value: string) => void
  onSelect: (id: string | null) => void
  saved?: SupplementPreview | null
}): React.JSX.Element {
  const {
    info,
    error,
    options,
    tolerance,
    disabled,
    preview: current,
    changeOptions,
    changeTolerance,
    changeAdditionalCharacters,
    choosePath,
    preparePreview
  } = useSupplementSettings({
    directory,
    busy,
    additionalCharacters,
    setAdditionalCharacters,
    onSelect
  })
  return (
    <section className="panel supplement-settings" aria-label="Train 문자 구성">
      <h3>Train 문자 구성</h3>
      <p>
        실제 train은 모두 유지합니다. 목표는 실제+합성 정답의 문자 수 비율이며 val/test는 가져온
        입력을 그대로 사용합니다.
      </p>
      {info && (
        <p className="field-help">
          실제 train {info.real.images}장 · 닉네임 {info.real.nicknames}개 · 정답 문자{' '}
          {info.real.characters}개 · 모델 사전 {info.dictionarySize}자
        </p>
      )}
      {info && (
        <details>
          <summary>실제 문자 빈도 보기</summary>
          <p className="character-frequencies">
            {info.real.frequencies
              .map((item) => `${JSON.stringify(item.character)} ${item.count}회`)
              .join(' · ')}
          </p>
        </details>
      )}
      <details className="training-runtime" open={Boolean(info?.missing.length)}>
        <summary>모델 사전 확장</summary>
        {info?.missing.length ? (
          <p role="status">
            현재 사전에 없는 실제 정답 문자:{' '}
            {info.missing.map((item) => `${item.character} (${item.count}회)`).join(' · ')}. 정답은
            그대로 보존됩니다.
          </p>
        ) : (
          <p className="field-help">
            현재 실제 정답은 모델 사전 안에 있습니다. 합성에 필요한 새 문자는 아래에 직접 추가할 수
            있습니다.
          </p>
        )}
        <label className="field">
          모델에 추가할 문자
          <textarea
            value={additionalCharacters}
            disabled={disabled}
            onChange={(event) => changeAdditionalCharacters(event.target.value)}
            placeholder="예: ★☆龍電あいうアイウ"
          />
        </label>
        <p className="field-help">
          입력한 새 문자는 기존 사전 뒤에 추가하고 해당 출력층을 새로 학습합니다. 기존 모델은
          유지하며 새 모델로 저장합니다. 사전 밖 실제 문자가 남아 있으면 학습할 수 없습니다.
        </p>
      </details>
      <label className="supplement-toggle">
        <input
          type="checkbox"
          checked={enabled}
          disabled={disabled}
          onChange={(event) => setEnabled(event.target.checked)}
        />{' '}
        부족한 문자군을 합성으로 보충
      </label>
      {enabled && (
        <>
          <SupplementDistribution
            info={info}
            options={options}
            current={current}
            disabled={disabled}
            onChange={changeOptions}
          />
          <div className="supplement-fields">
            <label className="field">
              허용 오차 (%p)
              <input
                type="number"
                min="0"
                max="100"
                step="any"
                value={tolerance}
                disabled={disabled}
                placeholder="직접 입력"
                onChange={(event) => changeTolerance(event.target.value)}
              />
            </label>
            <label className="field">
              합성 이미지 상한
              <input
                type="number"
                min="1"
                max="100000"
                value={options.maxImages}
                disabled={disabled}
                onChange={(event) => changeOptions({ maxImages: Number(event.target.value) })}
              />
            </label>
            <label className="field">
              합성 seed
              <input
                type="number"
                min="0"
                max="4294967295"
                value={options.seed}
                disabled={disabled}
                onChange={(event) => changeOptions({ seed: Number(event.target.value) })}
              />
            </label>
          </div>
          <SupplementAppearance
            options={options}
            disabled={disabled}
            onChange={changeOptions}
            onChoosePath={(kind) => void choosePath(kind)}
          />
          <button
            className="button button-secondary"
            disabled={disabled || !options.pythonExecutable || tolerance === ''}
            onClick={() => void preparePreview()}
          >
            보충량 계산·합성 미리보기
          </button>
          <p className="field-help">
            허용 오차 안에 들어오는 부족분을 계산합니다. 목표 미달이나 비율 오차는 경고로 표시하며,
            입력 오류와 구분합니다.
          </p>
          {current && (
            <div className="supplement-preview">
              <p role="status">
                계산한 부족분 {current.plan.requestedCharacters}자 · 생성 예상{' '}
                {current.plan.synthetic.images}장 · 문자 {current.plan.synthetic.characters}개 ·
                실제 train {current.plan.real.images}장 유지
              </p>
              {current.plan.warnings.map((warning) => (
                <p key={warning} className="field-help">
                  {warning}
                </p>
              ))}
              <div className="supplement-examples">
                {current.examples.map((example, index) => (
                  <figure key={index}>
                    <img src={example.image} alt={`합성 예시 ${example.text}`} />
                    <figcaption>{example.text}</figcaption>
                  </figure>
                ))}
              </div>
              <p>
                이 구성은 다음 학습 실행의 train에만 추가됩니다. 설정을 바꾸면 다시 미리보기하세요.
              </p>
            </div>
          )}
        </>
      )}
      {error && (
        <p role="alert" className="error-banner">
          {error}
        </p>
      )}
      {saved && saved.id === '' && (
        <details>
          <summary>저장된 실행의 합성 구성</summary>
          <p>
            합성 {saved.plan.synthetic.images}장 · 허용 오차 {saved.options.tolerance}%p · seed{' '}
            {saved.options.seed} · {saved.options.profile} · {saved.options.scaleModel} ·{' '}
            {saved.options.scaleSampling === 'fixed'
              ? `배율 ${saved.options.scale}`
              : '실제 train 높이 배율'}{' '}
            · {saved.options.placement === 'raid-name' ? '공대 이름 칸' : '가운데'}
          </p>
          <p>
            목표:{' '}
            {CHARACTER_GROUPS.filter((group) => saved.options.targets[group] !== null)
              .map((group) => `${CHARACTER_GROUP_LABELS[group]} ${saved.options.targets[group]}%`)
              .join(' · ')}
          </p>
          <p>모델 추가 문자: {saved.options.additionalCharacters || '없음'}</p>
          <div className="supplement-examples">
            {saved.examples.map((example, index) => (
              <figure key={index}>
                <img src={example.image} alt={`저장한 합성 예시 ${example.text}`} />
                <figcaption>{example.text}</figcaption>
              </figure>
            ))}
          </div>
        </details>
      )}
    </section>
  )
}
