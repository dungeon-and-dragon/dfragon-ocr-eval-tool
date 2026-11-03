import { mkdtemp, rm, writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { afterEach, expect, test } from 'vitest'
import {
  CHARACTER_GROUPS,
  characterDistribution,
  type SupplementOptions
} from '../../shared/supplement'
import { readSupplement } from './supplement'
import { parseSupplementOptions } from './supplement-validation'

const directories: string[] = []
afterEach(async () => {
  for (const directory of directories.splice(0)) {
    await rm(directory, { recursive: true, force: true })
  }
})

const options = (): SupplementOptions => ({
  additionalCharacters: '',
  pythonExecutable: join(tmpdir(), 'synth-python'),
  targets: Object.fromEntries(
    CHARACTER_GROUPS.map((group) => [group, group === 'latin' ? 50 : null])
  ) as SupplementOptions['targets'],
  characters: Object.fromEntries(
    CHARACTER_GROUPS.map((group) => [group, group === 'latin' ? 'AB' : ''])
  ) as SupplementOptions['characters'],
  tolerance: 0,
  maxImages: 10,
  seed: 1,
  profile: 'dotum',
  fonts: { gulim: join(tmpdir(), 'gulim.ttc'), batang: '', nanum: '', uttum: '' },
  scaleModel: 'calibrated-v1',
  scaleSampling: 'train-heights',
  placement: 'raid-name',
  exactUnitScaleShare: 0.5,
  scale: 0.5,
  width: 160,
  height: 32,
  padding: 0,
  color: [214, 197, 146],
  backgroundMode: 'solid',
  backgroundColor: [0, 0, 0],
  backgroundDirectory: ''
})

test('합성 설정은 확대 방식, 배율 분포, 글자 배치를 그대로 넘기고 모르는 값은 거절한다', () => {
  expect(parseSupplementOptions(options())).toMatchObject({
    scaleModel: 'calibrated-v1',
    scaleSampling: 'train-heights',
    placement: 'raid-name'
  })
  expect(() => parseSupplementOptions({ ...options(), exactUnitScaleShare: 1.5 })).toThrow(
    '숫자 범위'
  )
  for (const change of [
    { scaleModel: 'nearest' },
    { scaleSampling: 'random' },
    { placement: undefined }
  ]) {
    expect(() => parseSupplementOptions({ ...options(), ...change })).toThrow('확대 방식')
  }
})

test('calibrated-v1의 고정 배율은 1 미만을 거절하고 bilinear는 받는다', () => {
  expect(() => parseSupplementOptions({ ...options(), scaleSampling: 'fixed' })).toThrow(
    'calibrated-v1'
  )
  expect(
    parseSupplementOptions({ ...options(), scaleSampling: 'fixed', scaleModel: 'affine-bilinear' })
      .scale
  ).toBe(0.5)
})

test('옵션이 생기기 전 저장된 합성 결과는 bilinear 고정 배율 가운데 배치로 읽는다', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'supplement-'))
  directories.push(directory)
  const legacy: Partial<SupplementOptions> = { ...options() }
  delete legacy.scaleModel
  delete legacy.scaleSampling
  delete legacy.placement
  delete legacy.exactUnitScaleShare
  const plan = {
    real: characterDistribution(['가']),
    synthetic: characterDistribution(['AB']),
    final: characterDistribution(['가', 'AB']),
    differences: Object.fromEntries(
      CHARACTER_GROUPS.map((group) => [group, group === 'latin' ? 0 : null])
    ),
    requestedCharacters: 2,
    warnings: []
  }
  const manifest = (schemaVersion: number, value: unknown) =>
    JSON.stringify({ schemaVersion, options: value, plan, samples: [] })

  await writeFile(join(directory, 'supplement.json'), manifest(1, legacy))
  expect((await readSupplement(directory, directory, false)).options).toMatchObject({
    scaleModel: 'affine-bilinear',
    scaleSampling: 'fixed',
    placement: 'center'
  })
  await writeFile(join(directory, 'supplement.json'), manifest(2, legacy))
  await expect(readSupplement(directory, directory, false)).rejects.toThrow('확대 방식')
})
