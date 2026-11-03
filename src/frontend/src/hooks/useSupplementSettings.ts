import { useEffect, useState } from 'react'
import {
  CHARACTER_GROUPS,
  SUPPLEMENT_CROP_SIZE,
  type SupplementInfo,
  type SupplementOptions,
  type SupplementPreview
} from '../../../shared/supplement'

const emptyTargets = () =>
  Object.fromEntries(CHARACTER_GROUPS.map((group) => [group, null])) as SupplementOptions['targets']
const emptyCharacters = () =>
  Object.fromEntries(
    CHARACTER_GROUPS.map((group) => [group, ''])
  ) as SupplementOptions['characters']

export function useSupplementSettings({
  directory,
  busy,
  additionalCharacters,
  setAdditionalCharacters,
  onSelect
}: {
  directory: string
  busy: boolean
  additionalCharacters: string
  setAdditionalCharacters: (value: string) => void
  onSelect: (id: string | null) => void
}) {
  const [info, setInfo] = useState<SupplementInfo | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [preview, setPreview] = useState<SupplementPreview | null>(null)
  const [pending, setPending] = useState(false)
  const [tolerance, setTolerance] = useState('')
  const [options, setOptions] = useState<SupplementOptions>({
    additionalCharacters,
    pythonExecutable: '',
    targets: emptyTargets(),
    characters: emptyCharacters(),
    tolerance: 0,
    maxImages: 10000,
    seed: 42,
    profile: 'dotum',
    fonts: {
      gulim: 'C:/Windows/Fonts/gulim.ttc',
      batang: 'C:/Windows/Fonts/batang.ttc',
      nanum: '',
      uttum: ''
    },
    scaleModel: 'calibrated-v1',
    scaleSampling: 'train-heights',
    placement: 'raid-name',
    scale: 1,
    width: 160,
    height: 32,
    padding: 0,
    color: [214, 197, 146],
    backgroundMode: 'solid',
    backgroundColor: [24, 28, 32],
    backgroundDirectory: ''
  })
  const disabled = busy || pending
  useEffect(() => {
    let disposed = false
    void window.training
      .supplementInfo()
      .then((result) => {
        if (disposed) {
          return
        }
        if (result.ok) {
          setInfo(result.value)
          setOptions((value) => ({
            ...value,
            width: Math.min(
              SUPPLEMENT_CROP_SIZE.max,
              Math.max(SUPPLEMENT_CROP_SIZE.min, result.value.width)
            ),
            height: Math.min(
              SUPPLEMENT_CROP_SIZE.max,
              Math.max(SUPPLEMENT_CROP_SIZE.min, result.value.height)
            )
          }))
        } else {
          setError(result.error)
        }
      })
      .catch(() => {
        if (!disposed) {
          setError('실제 문자 분포를 읽지 못했습니다.')
        }
      })
    return () => {
      disposed = true
    }
  }, [directory])

  function invalidatePreview(): void {
    setPreview(null)
    onSelect(null)
  }
  function change(next: Partial<SupplementOptions>): void {
    setOptions((value) => ({ ...value, ...next }))
    invalidatePreview()
  }
  async function choose(kind: 'python' | 'output'): Promise<void> {
    const result = await window.evaluation.choosePath(kind)
    if (!result.ok) {
      setError(result.error)
    } else if (result.value) {
      change(
        kind === 'python'
          ? { pythonExecutable: result.value }
          : { backgroundDirectory: result.value }
      )
    }
  }
  async function prepare(): Promise<void> {
    setPending(true)
    setError(null)
    invalidatePreview()
    try {
      const result = await window.training.previewSupplement({
        ...options,
        additionalCharacters,
        tolerance: Number(tolerance)
      })
      if (!result.ok) {
        setError(result.error)
      } else {
        setPreview(result.value)
        onSelect(result.value.id)
      }
    } catch {
      setError('합성 미리보기를 준비하지 못했습니다.')
    } finally {
      setPending(false)
    }
  }
  return {
    info,
    error,
    options,
    tolerance,
    disabled,
    preview: preview?.options.additionalCharacters === additionalCharacters ? preview : null,
    changeOptions: change,
    changeTolerance(value: string) {
      setTolerance(value)
      invalidatePreview()
    },
    changeAdditionalCharacters(value: string) {
      setAdditionalCharacters(value)
      invalidatePreview()
    },
    choosePath: choose,
    preparePreview: prepare
  }
}
