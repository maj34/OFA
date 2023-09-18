# OFA
Official repository of OFA (ICML 2022). Paper: OFA: Unifying Architectures, Tasks, and Modalities Through a Simple Sequence-to-Sequence Learning Framework

#### 논문 링크

[OFA: Unifying Architectures, Tasks, and Modalities Through a...](https://arxiv.org/abs/2202.03052v2)

#### 논문 리뷰

[OFA: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework (ICML 2022)](https://www.notion.so/OFA-Unifying-architectures-tasks-and-modalities-through-a-simple-sequence-to-sequence-learning-fr-412ef4c30bc14277aae24cca1d94211a?pvs=21)

<br/>

# OFA 모델 소개

![BLEU-4 Score 44.9 (2022 SOTA)](https://prod-files-secure.s3.us-west-2.amazonaws.com/5374ba22-7d52-4910-9324-cd5d12d3dd78/f450be41-2d76-4f9b-be2a-c32d409ea0f3/Untitled.png)

BLEU-4 Score 44.9 (2022 SOTA)

![Accuracy 79.36 test-dev / Accuracy 79.48 test-std](https://prod-files-secure.s3.us-west-2.amazonaws.com/5374ba22-7d52-4910-9324-cd5d12d3dd78/2b890749-8dff-4da7-bf18-f947e9538899/Untitled.png)

Accuracy 79.36 test-dev / Accuracy 79.48 test-std

![Accuracy 91.2 test (SOTA) / Accuracy 91.0 val (SOTA)](https://prod-files-secure.s3.us-west-2.amazonaws.com/5374ba22-7d52-4910-9324-cd5d12d3dd78/c22fc624-ccc2-4bc1-adb0-395350856e94/Untitled.png)

Accuracy 91.2 test (SOTA) / Accuracy 91.0 val (SOTA)

---

- multimodal pretraining을 위한 통합된 패러다임을 구축함
- **OFA(One-for-All)**라는 unified multimodal pretrained model을 제시
- OFA는 **단순한 sequence-to-sequence 학습 프레임워크를 기반**으로 하고 인코더-디코더 구조를 사용
- OFA는 **fine-tuning시 task-specific한 레이어를 요구하지도 않음**
- OFA는 다양한 **multimodal task에서 SOTA를 달성**함은 물론이고 unimodal task 에서조차 
**unimodal 만을 위해 학습된 모델들과 동등한 수준의 성능**을 보여줌

![Untitled](https://prod-files-secure.s3.us-west-2.amazonaws.com/5374ba22-7d52-4910-9324-cd5d12d3dd78/f88575f0-c026-493c-9084-1f5df660eee6/Untitled.png)

**1. Task-Agnostic(TA)**: 분류, 생성, self-supervised pretext task 등의 **다양한 과제에 공통적으로 사용**할 수 있는 unified task representation를 활용해야 함

**2. Modality-Agnostic(MA)**: **여러 가지 양식(modality)**를 다룰 수 있는 unified input and output representation

**3. Task Comprehensiveness(TC)**: generalization 능력을 얻을 수 있을 만큼의 **과제(task) 다양성**

# 결과 분석

### Evaluation 실험

---

- val_dataset (10,402개)
    - batch_size
    - 한 번 학습할 때 얼마나 많은 문장(max_sentences)을 학습할 것인지 의미
    
    ![Accuracy of Different Batch Size.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/5374ba22-7d52-4910-9324-cd5d12d3dd78/d5b7a77f-508e-404c-b3a6-3bf21a8d586f/Accuracy_of_Different_Batch_Size.png)

### Inference Pipeline 구축

---

```python
import re
from PIL import Image

def inference(model_path, image_path):
    # specify some options for evaluation
    parser = options.get_generation_parser()
    input_args = ["", "--task=vqa_gen", "--beam=100", "--unnormalized", f"--path=checkpoints/{model_path}.pt", "--bpe-dir=utils/BPE"]
    args = options.parse_args_and_arch(parser, input_args)
    cfg = convert_namespace_to_omegaconf(args)

    # Load pretrained ckpt & config
    task = tasks.setup_task(cfg.task)
    models, cfg = checkpoint_utils.load_model_ensemble(
        utils.split_paths(cfg.common_eval.path),
        task=task
    )

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    image = Image.open(f'./input_images/{image_path}')
    questions = [
                "Where is the person?",
                "How many people are in the picture?",
                "How is the weather like in the picture?",
                "What accessories does the person wearing?",
                "What kind of hairstyle does the person have?",
                "What kind of shirt does a person wearing?",
                "How old does the person look like exactly?",
                "What country do you look like?"
                ]

    display(image)

    for idx, question in enumerate(questions):
        sample = construct_sample(image, question)
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

        # Run eval step for open-domain VQA
        with torch.no_grad():
            result, scores = zero_shot_step(task, generator, models, sample)

        print(f"Q{idx + 1}: {question}")
        print(f"A{idx + 1}: {result[0]['answer']}")
        print()
```

- ofa_tiny, ofa_medium, ofa_base, ofa_large, ofa_huge 5개의 모델 실험
    - 모델의 크기가 커질수록 정확한 답을 하며 주관적인 질문도 그럴듯하게 잘 답변하는 것을 확인할 수 있음
    - 에니메이션 사진, 합성 사진 등 out-of-domain input에 대해서도 잘 작동함
- 정성적인 결과는 [demo.py](http://demo.py) 파일 참고
