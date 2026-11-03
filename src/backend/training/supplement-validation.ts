import {
  CHARACTER_GROUPS,
  PLACEMENTS,
  SCALE_MODELS,
  SCALE_SAMPLINGS,
  SUPPLEMENT_CROP_SIZE,
  characterGroup,
  type CharacterDistribution,
  type SupplementOptions,
  type SupplementPlan
} from '../../shared/supplement'
import { isRecord, readPath } from '../evaluation/validation'

function isDistribution(value: unknown): value is CharacterDistribution {
  if (!isRecord(value) || !isRecord(value.groups) || !Array.isArray(value.frequencies)) {
    return false
  }
  const groupCounts = value.groups
  const count = (n: unknown): n is number => Number.isSafeInteger(n) && Number(n) >= 0
  if (
    !count(value.images) ||
    !count(value.nicknames) ||
    !count(value.characters) ||
    value.nicknames > value.images ||
    CHARACTER_GROUPS.some((group) => !count(groupCounts[group]))
  ) {
    return false
  }
  const seen = new Set<string>()
  const groups = Object.fromEntries(CHARACTER_GROUPS.map((group) => [group, 0]))
  for (const row of value.frequencies) {
    if (
      !isRecord(row) ||
      typeof row.character !== 'string' ||
      [...row.character].length !== 1 ||
      !count(row.count) ||
      row.count === 0 ||
      row.group !== characterGroup(row.character) ||
      seen.has(row.character)
    ) {
      return false
    }
    seen.add(row.character)
    groups[characterGroup(row.character)] += row.count
  }
  return (
    Object.values(groups).reduce((sum, n) => sum + n, 0) === value.characters &&
    CHARACTER_GROUPS.every((group) => groups[group] === groupCounts[group])
  )
}

export function isSupplementPlan(value: unknown): value is SupplementPlan {
  if (!isRecord(value) || !isRecord(value.differences)) {
    return false
  }
  const differences = value.differences
  return (
    isDistribution(value.real) &&
    isDistribution(value.synthetic) &&
    isDistribution(value.final) &&
    CHARACTER_GROUPS.every((group) => {
      const delta = differences[group]
      return (
        delta === null ||
        (typeof delta === 'number' && Number.isFinite(delta) && Math.abs(delta) <= 100)
      )
    }) &&
    Number.isSafeInteger(value.requestedCharacters) &&
    Number(value.requestedCharacters) >= 0 &&
    Array.isArray(value.warnings) &&
    value.warnings.every((warning) => typeof warning === 'string') &&
    value.final.images === value.real.images + value.synthetic.images &&
    value.final.characters === value.real.characters + value.synthetic.characters
  )
}

export function parseSupplementOptions(value: unknown): SupplementOptions {
  if (
    !isRecord(value) ||
    !isRecord(value.targets) ||
    !isRecord(value.characters) ||
    !isRecord(value.fonts)
  ) {
    throw new Error('합성 설정을 확인해 주세요.')
  }
  const number = (input: unknown, min: number, max: number, integer = false): number => {
    if (
      typeof input !== 'number' ||
      !Number.isFinite(input) ||
      input < min ||
      input > max ||
      (integer && !Number.isInteger(input))
    ) {
      throw new Error('합성 설정의 숫자 범위를 확인해 주세요.')
    }
    return input
  }
  const rgb = (input: unknown) => {
    if (!Array.isArray(input) || input.length !== 3) {
      throw new Error('RGB 색상을 확인해 주세요.')
    }
    return input.map((channel) => number(channel, 0, 255, true))
  }
  if (
    typeof value.additionalCharacters !== 'string' ||
    value.additionalCharacters.length > 20_000
  ) {
    throw new Error('모델에 추가할 문자를 확인해 주세요.')
  }
  const targets = {} as SupplementOptions['targets']
  const characters = {} as SupplementOptions['characters']
  for (const group of CHARACTER_GROUPS) {
    targets[group] = value.targets[group] === null ? null : number(value.targets[group], 0, 100)
    const text = value.characters[group]
    if (typeof text !== 'string' || text.length > 20_000) {
      throw new Error('합성 문자 목록을 확인해 주세요.')
    }
    characters[group] = text
  }
  if (Object.values(targets).reduce<number>((sum, target) => sum + (target ?? 0), 0) > 100 + 1e-8) {
    throw new Error('목표 비율 합계가 100%를 넘습니다.')
  }
  if (
    (value.profile !== 'dotum' && value.profile !== 'nanum-neo') ||
    (value.backgroundMode !== 'solid' && value.backgroundMode !== 'images')
  ) {
    throw new Error('글꼴 모드·배경 종류를 확인해 주세요.')
  }
  const oneOf = <T extends string>(choices: readonly T[], input: unknown): T => {
    const choice = choices.find((item) => item === input)
    if (choice === undefined) {
      throw new Error('확대 방식·배율 분포·글자 배치를 확인해 주세요.')
    }

    return choice
  }
  const scaleModel = oneOf(SCALE_MODELS, value.scaleModel)
  const scaleSampling = oneOf(SCALE_SAMPLINGS, value.scaleSampling)
  const scale = number(value.scale, Number.MIN_VALUE, 16)
  if (scaleModel === 'calibrated-v1' && scaleSampling === 'fixed' && scale < 1) {
    throw new Error('calibrated-v1은 배율 1 이상만 지원합니다.')
  }
  const fonts = {} as SupplementOptions['fonts']
  for (const name of ['gulim', 'batang', 'nanum', 'uttum'] as const) {
    fonts[name] = value.fonts[name] === '' ? '' : readPath(value.fonts[name])
  }
  const width = number(value.width, SUPPLEMENT_CROP_SIZE.min, SUPPLEMENT_CROP_SIZE.max, true)
  const height = number(value.height, SUPPLEMENT_CROP_SIZE.min, SUPPLEMENT_CROP_SIZE.max, true)
  return {
    additionalCharacters: value.additionalCharacters,
    pythonExecutable: readPath(value.pythonExecutable),
    targets,
    characters,
    tolerance: number(value.tolerance, 0, 100),
    maxImages: number(value.maxImages, 1, 100_000, true),
    seed: number(value.seed, 0, 2 ** 32 - 1, true),
    profile: value.profile,
    fonts,
    scaleModel,
    scaleSampling,
    placement: oneOf(PLACEMENTS, value.placement),
    scale,
    width,
    height,
    padding: number(value.padding, 0, Math.floor((Math.min(width, height) - 1) / 2), true),
    color: rgb(value.color),
    backgroundMode: value.backgroundMode,
    backgroundColor: rgb(value.backgroundColor),
    backgroundDirectory:
      value.backgroundMode === 'images' ? readPath(value.backgroundDirectory) : ''
  }
}
