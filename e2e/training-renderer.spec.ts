import { expect, test } from '@playwright/test'
import { installRendererFixture } from './renderer-fixture'
import type { TrainingSnapshot } from '../src/shared/training'

test('library selection leads to local training, test review and explicit publication only', async ({
  page
}, info) => {
  const errors: string[] = []
  page.on('pageerror', (error) => errors.push(error.message))
  await installRendererFixture(page, 'completed')
  await page.goto('/')
  await page.evaluate(() => {
    const model = {
      id: '00000000-0000-4000-8000-000000000001',
      name: '한국어 PP-OCRv5 · 기본 모델',
      kind: 'pretrained' as const,
      preset: 'korean-ppocrv5' as const,
      parentId: null,
      registeredAt: '2026-09-27T00:00:00Z',
      files: []
    }
    let state: TrainingSnapshot = {
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
    let listener: (state: TrainingSnapshot) => void = () => undefined
    const fixture = window as unknown as { trainingPublishes: number; finishTraining: () => void }
    fixture.trainingPublishes = 0
    const update = (next: Partial<TrainingSnapshot>) => {
      state = { ...state, ...next }
      listener(state)
    }
    window.evaluation.choosePath = async () => ({ ok: true, value: 'C:\\fixture\\experiments' })
    window.training = {
      login: async () => ({ ok: true, value: null }),
      logout: async () => ({ ok: true, value: null }),
      models: async () => ({ ok: true, value: [model] }),
      defaults: async () => ({
        ok: true,
        value: {
          pythonExecutable: 'C:\\fixture\\python.exe',
          upstreamDirectory: 'C:\\fixture\\PaddleOCR'
        }
      }),
      download: async () => {
        update({
          status: 'ready',
          model,
          directory: 'C:\\fixture\\experiments\\experiment-demo',
          counts: { train: 30, val: 8, test: 1, skipped: 4 },
          message: '모델과 데이터를 고정했습니다.'
        })
        return { ok: true, value: null }
      },
      supplementInfo: async () => ({
        ok: true,
        value: {
          real: {
            images: 30,
            nicknames: 25,
            characters: 90,
            groups: {
              hangul: 90,
              special: 0,
              hiragana: 0,
              katakana: 0,
              hanja: 0,
              latin: 0,
              digit: 0,
              other: 0
            },
            frequencies: []
          },
          missing: [],
          dictionarySize: 11945,
          width: 8192,
          height: 4096
        }
      }),
      previewSupplement: async () => ({ ok: false, error: 'fixture: configure synthesis' }),
      open: async () => ({ ok: true, value: null }),
      start: async () => {
        update({ status: 'training', epoch: 1, epochs: 10, message: 'train으로 학습 중입니다.' })
        return { ok: true, value: null }
      },
      cancel: async () => {
        update({ status: 'cancelled', message: '취소했습니다.' })
        return { ok: true, value: null }
      },
      publish: async () => {
        fixture.trainingPublishes++
        update({ publishedModelId: 'registered', message: '미니PC에 등록했습니다.' })
        return { ok: true, value: model }
      },
      readImage: async () => ({
        ok: true,
        value:
          'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/L9sAAAAASUVORK5CYII='
      }),
      getSnapshot: async () => ({ ok: true, value: state }),
      onSnapshot: (next) => {
        listener = next
        return () => undefined
      }
    }
    fixture.finishTraining = () =>
      update({
        status: 'completed',
        epoch: 10,
        message: '학습·test 평가 완료',
        metrics: { cer: 1 / 3, exactMatch: 0, sampleCount: 1, characterCount: 3 },
        samples: [
          {
            id: 'fixture-test',
            imagePath: 'C:\\fixture\\test.png',
            truth: '새벽꽃',
            prediction: '새벽꼿',
            editDistance: 1,
            confidence: 0.94
          }
        ]
      })
  })
  await page.getByRole('tab', { name: '학습', exact: true }).click()
  await page.getByRole('button', { name: '자료실 로그인' }).click()
  await expect(page.getByLabel('시작 모델')).toHaveValue('00000000-0000-4000-8000-000000000001')
  await page.getByRole('button', { name: '실험 저장 폴더 선택' }).click()
  await page.getByRole('button', { name: '모델·데이터 가져오기' }).click()
  await expect(page.getByText('train 30 · val 8 · test 1 · 사용 안 함 4')).toBeVisible()
  await page.getByLabel('부족한 문자군을 합성으로 보충').check()
  await page.getByLabel('글자 배치').selectOption('center')
  await expect(page.getByLabel('크롭 너비 (px)')).toHaveValue('2048')
  await expect(page.getByLabel('크롭 높이 (px)')).toHaveValue('2048')
  await page.getByLabel('크롭 너비 (px)').fill('160')
  await page.getByLabel('크롭 높이 (px)').fill('32')
  await expect(page.getByRole('button', { name: '학습 후 test 평가' })).toBeDisabled()
  await expect(page.getByLabel('특수문자 목표 (%)')).toHaveValue('')
  await expect(page.getByLabel('허용 오차 (%p)')).toHaveValue('')
  await page.getByLabel('특수문자 목표 (%)').fill('10')
  await page.getByLabel('특수문자 합성 문자').fill('★☆')
  await page.getByLabel('허용 오차 (%p)').fill('1')
  await page.getByText('모델 사전 확장', { exact: true }).click()
  await page.getByLabel('모델에 추가할 문자').fill('★☆')
  await page.getByRole('button', { name: '합성용 Python 선택' }).click()
  await page.evaluate(() => {
    window.training.previewSupplement = async (options) => {
      const info = await window.training.supplementInfo()
      if (!info.ok) {
        return info
      }
      const real = info.value.real
      return {
        ok: true,
        value: {
          id: '00000000-0000-4000-8000-000000000002',
          options,
          plan: {
            real,
            synthetic: {
              ...real,
              images: 1,
              nicknames: 1,
              characters: 2,
              groups: { ...real.groups, hangul: 0, special: 2 }
            },
            final: {
              ...real,
              images: 31,
              nicknames: 26,
              characters: 92,
              groups: { ...real.groups, special: 2 }
            },
            differences: {
              hangul: null,
              special: -7.83,
              hiragana: null,
              katakana: null,
              hanja: null,
              latin: null,
              digit: null,
              other: null
            },
            requestedCharacters: 9,
            warnings: ['목표 비율 미달 · 이 구성으로 실행할 수 있습니다.']
          },
          examples: [
            {
              text: '★☆',
              image:
                'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/L9sAAAAASUVORK5CYII='
            }
          ]
        }
      }
    }
  })
  await page.getByRole('button', { name: '보충량 계산·합성 미리보기' }).click()
  await expect(page.getByText('목표 비율 미달 · 이 구성으로 실행할 수 있습니다.')).toBeVisible()
  await expect(page.getByRole('button', { name: '학습 후 test 평가' })).toBeEnabled()
  await page.screenshot({ path: info.outputPath('synthetic-preview.png'), fullPage: true })
  await page.getByLabel('특수문자 목표 (%)').fill('12')
  await expect(page.getByRole('button', { name: '학습 후 test 평가' })).toBeDisabled()
  await page.getByRole('button', { name: '보충량 계산·합성 미리보기' }).click()
  await page.getByRole('button', { name: '학습 후 test 평가' }).click()
  await expect(page.getByRole('button', { name: '모델·데이터 가져오기' })).toBeDisabled()
  await expect(page.getByRole('region', { name: '학습 모델의 test 성적' })).not.toContainText('%')
  await expect(page.getByRole('button', { name: '미니PC에 올리기' })).toHaveCount(0)
  await page.evaluate(() => (window as unknown as { finishTraining: () => void }).finishTraining())
  await expect(page.getByRole('region', { name: '학습 모델의 test 성적' })).toContainText('33.33%')
  await expect(page.getByText('새벽꼿', { exact: true })).toBeVisible()
  expect(
    await page.evaluate(
      () => (window as unknown as { trainingPublishes: number }).trainingPublishes
    )
  ).toBe(0)
  await page.getByLabel('새 모델 이름').fill('닉네임 모델 · 1차 학습')
  await page.screenshot({ path: info.outputPath('training-review.png'), fullPage: true })
  await page.getByRole('button', { name: '미니PC에 올리기' }).click()
  expect(
    await page.evaluate(
      () => (window as unknown as { trainingPublishes: number }).trainingPublishes
    )
  ).toBe(1)
  await expect(page.getByRole('button', { name: '자료실 등록 완료' })).toBeDisabled()
  expect(errors).toEqual([])
})
