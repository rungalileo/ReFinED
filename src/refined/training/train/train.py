import os
import sys

sys.path.append("src")


from typing import List

import torch
from torch.cuda.amp import GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from refined.data_types.doc_types import Doc
from refined.dataset_reading.entity_linking.wikipedia_dataset import WikipediaDataset
from refined.doc_preprocessing.preprocessor import PreprocessorInferenceOnly
from refined.doc_preprocessing.wikidata_mapper import WikidataMapper
from refined.inference.processor import Refined
from refined.model_components.config import NER_TAG_TO_IX, ModelConfig
from refined.model_components.refined_model import RefinedModel
from refined.resource_management.aws import S3Manager
from refined.resource_management.resource_manager import ResourceManager
from refined.torch_overrides.data_parallel_refined import DataParallelReFinED
from refined.training.fine_tune.fine_tune import run_fine_tuning_loops
from refined.training.train.training_args import parse_training_args
from refined.utilities.general_utils import get_logger
from refined.utilities.galileo_helpers import log_input_data_galileo

import dataquality as dq

LOG = get_logger(name=__name__)


# def get_span_context(text, span_start, span_len):
#     pre_span = text[:span_start]
#     sentance_start = max(
#         pre_span.rfind("."),
#         pre_span.rfind("?"),
#         pre_span.rfind("!"),
#         pre_span.rfind("\n"),
#     ) + 1
#     if sentance_start != 0: sentance_start += 1
#
#     sentance_end = len(text) - 1
#     for punc in ["?", "!", ".", "\n"]:
#         end_punctuation = text.find(punc, span_start + span_len)
#         if end_punctuation != -1:
#             sentance_end = min(sentance_end, end_punctuation)
#
#     sentance_end += 1
#
#     span_text_with_context = f"{text[sentance_start:span_start]}<<{text[span_start: span_start+span_len]}>>{text[span_start+span_len: sentance_end]}"
#     return span_text_with_context
#
#
# def log_input_data_galileo(
#         dataset: WikipediaDataset, preprocessor: PreprocessorInferenceOnly, split: str, max_batch_size: int
# ) -> None:
#     """Log entity type data with Galileo as MLTC data
#
#     ...
#     """
#     lookups = preprocessor.lookups
#     # Create map from idx --> entity type name. Used for logging readable labels
#     # Shift every by one to allow for the dummy 0th class
#     idx_to_label_name = {idx: lookups.class_to_label[lookups.index_to_class[idx]] for idx in lookups.index_to_class.keys()}
#     # 🔭🌕 Galileo logging
#     # Ensure that the logged tasks match the
#     # task_ids_sorted = sorted(idx_to_label_name.keys())
#     # Just log the tasks we are interested in!!
#     task_ids_sorted = lookups.label_subset_arr
#     tasks = [str(idx_to_label_name[idx]) for idx in list(task_ids_sorted)]
#     import json
#     with open("tasks.json", 'w') as f:
#         json.dump(tasks, f)
#     # TODO DON'T NEED to Add Q0 as the non-task label since we can filter this out.
#     # dq.set_tasks_for_run(["Q0"] + tasks, binary=True)
#     dq.set_tasks_for_run(tasks, binary=True)
#
#     # Loop over the BatchElementTns
#     span_texts = []
#     spans = []
#     span_labels = []
#     span_ids = []
#     meta_data = {"doc_id": [], "is_md_span": []}
#     # Why convert to a list?
#     for data in dataset:
#         # Handle the Validation dataset case where we have must convert documents first
#         # to a list of BatchedElementsTns
#         batches: List[BatchedElementsTns] = []
#         if type(data) == Doc:
#             # type(batch) == DOC -> convert it to a list of BatchedElementsTns.
#             for batch_doc_tns in convert_doc_to_tensors(
#                 data,
#                 preprocessor,
#                 collate=True,
#                 max_batch_size=max_batch_size,
#                 sort_by_tokens=False,
#                 max_seq=preprocessor.max_seq,
#             ):
#                 batches.append(batch_doc_tns)
#         else:
#             batches = [data]
#
#         for batch in batches:
#             for i, batch_element in enumerate(batch.batch_elements):
#                 for j, span in enumerate(batch_element.spans):
#                     labels = batch.class_target_values[i][j]
#                     labels = labels[labels != 0].numpy()
#                     # TODO Don't Need Dummy label
#                     # labels = ["Q0"] + [str(idx_to_label_name[idx]) for idx in labels]
#                     # TODO Only log labels we are watching
#                     labels = [str(idx_to_label_name[idx]) for idx in labels if idx in preprocessor.lookups.label_subset_set]
#                     if len(labels) == 0:
#                         continue
#
#                     span_text = get_span_context(batch_element.text, span.start, span.ln)
#
#                     span_texts.append(span_text)
#                     spans.append(span.text)
#                     span_labels.append(labels)
#                     span_ids.append(batch_element.span_ids[j])
#
#                     meta_data['doc_id'].append(span.doc_id)
#                     meta_data['is_md_span'].append(span.is_md_span)
#
#
#     import pandas as pd
#     df = pd.DataFrame({"text": span_texts, "labels": span_labels, "id": span_ids, "doc_id": meta_data["doc_id"], "span_text": spans})
#     df.to_csv("Musixmatch_Labels_500-550_tiny.csv")
#
#     # TODO It may be very intensive to store all of these. May need to incrementally log
#     # 🔭🌕 Galileo logging
#     dq.log_data_samples(
#         texts=span_texts,
#         task_labels=span_labels,
#         ids=span_ids,
#         split=split
#     )


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["GALILEO_CONSOLE_URL"] = "https://console.sandbox.rungalileo.io"
    os.environ["GALILEO_USERNAME"] = "galileo@rungalileo.io"
    os.environ["GALILEO_PASSWORD"] = "Th3secret_"

    # 🔭🌕 Galileo logging
    dq.configure()
    dq.init(task_type="text_multi_label",
            project_name="Testing_ED",
            run_name=f"tiny")

    # DDP (ensure batch_elements_included is used)

    training_args = parse_training_args()

    resource_manager = ResourceManager(S3Manager(),
                                       data_dir=training_args.data_dir,
                                       entity_set=training_args.entity_set,
                                       load_qcode_to_title=True,
                                       load_descriptions_tns=True,
                                       model_name=None
                                       )
    if training_args.download_files:
        resource_manager.download_data_if_needed()
        resource_manager.download_additional_files_if_needed()
        resource_manager.download_training_files_if_needed()

    preprocessor = PreprocessorInferenceOnly(
        data_dir=training_args.data_dir,
        debug=training_args.debug,
        max_candidates=training_args.num_candidates_train,
        transformer_name=training_args.transformer_name,
        ner_tag_to_ix=NER_TAG_TO_IX,  # for now include default ner_to_tag_ix can make configurable in future
        entity_set=training_args.entity_set,
        use_precomputed_description_embeddings=False,
    )

    wikidata_mapper = WikidataMapper(resource_manager=resource_manager)

    wikipedia_dataset_file_path = resource_manager.get_training_data_files()['wikipedia_training_dataset']
    training_dataset = WikipediaDataset(
        # start=100,
        start=100,
        end=500,  #100000000, 150  # large number means every line will be read until the end of the file
        preprocessor=preprocessor,
        resource_manager=resource_manager,
        wikidata_mapper=wikidata_mapper,
        dataset_path=wikipedia_dataset_file_path,
        batch_size=training_args.batch_size,
        num_workers=1, # 8 * training_args.n_gpu,
        prefetch=100,  # add random number for each worker and have more than 2 workers to remove waiting
        mask=training_args.mask_prob,
        random_mask=training_args.mask_random_prob,
        lower_case_prob=0.05,
        candidate_dropout=training_args.candidate_dropout,
        max_mentions=training_args.max_mentions,
        sample_k_candidates=5,
        add_main_entity=True,
    )


    training_dataloader = DataLoader(dataset=training_dataset, batch_size=None, # num_workers=8 * training_args.n_gpu,
                                     # pin_memory=True if training_args.n_gpu == 1 else False,
                                     #pin_memory=True,  # may break ddp and dp training
                                     #prefetch_factor=5,  # num_workers * prefetch_factor
                                     #persistent_workers=True  # persistent_workers means memory is stable across epochs
                                     )

    eval_dataset = WikipediaDataset(
        start=0,
        end=10,  # first 100 docs are used for eval
        preprocessor=preprocessor,
        resource_manager=resource_manager,
        wikidata_mapper=wikidata_mapper,
        dataset_path=wikipedia_dataset_file_path,
        return_docs=True,  # this means the dataset will return `Doc` objects instead of BatchedElementsTns
        batch_size=1 * training_args.n_gpu,
        num_workers=1,
        prefetch=1,
        mask=0.0,
        random_mask=0.0,
        lower_case_prob=0.0,
        candidate_dropout=0.0,
        sample_k_candidates=5,
        max_mentions=25,  # prevents memory issues
        add_main_entity=True  # add weak labels,!
    )
    eval_docs: List[Doc] = list(iter(eval_dataset))

    # Log training + validation data
    print ("Logging Training Data")
    log_input_data_galileo(training_dataset, preprocessor, "training", training_args.batch_size)
    print("Logging Validation Data")
    log_input_data_galileo(eval_dataset, preprocessor, "validation", training_args.batch_size)

    model = RefinedModel(
        ModelConfig(data_dir=preprocessor.data_dir,
                    transformer_name=preprocessor.transformer_name,
                    ner_tag_to_ix=preprocessor.ner_tag_to_ix
                    ),
        preprocessor=preprocessor
    )

    if training_args.restore_model_path is not None:
        # TODO load `ModelConfig` file (from the directory) and initialise RefinedModel from that
        # to avoid issues when model config differs
        LOG.info(f'Restored model from {training_args.restore_model_path}')
        checkpoint = torch.load(training_args.restore_model_path, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)

    if training_args.n_gpu > 1:
        model = DataParallelReFinED(model, device_ids=list(range(training_args.n_gpu)), output_device=training_args.device)
    model = model.to(training_args.device)

    # wrap a ReFinED processor around the model so evaluation methods can be run easily
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )
    refined = Refined(
        model_file_or_model=model,
        model_config_file_or_model_config=model_to_save.config,
        preprocessor=preprocessor,
        device=training_args.device
    )

    param_groups = [
        {"params": model_to_save.get_et_params(), "lr": training_args.lr * 100},
        {"params": model_to_save.get_desc_params(), "lr": training_args.lr},
        {"params": model_to_save.get_ed_params(), "lr": training_args.lr * 100},
        {"params": model_to_save.get_parameters_not_to_scale(), "lr": training_args.lr}
    ]
    if training_args.el:
        param_groups.append({"params": model_to_save.get_md_params(), "lr": training_args.lr})

    optimizer = AdamW(param_groups, lr=training_args.lr, eps=1e-8)

    total_steps = len(training_dataloader) * training_args.epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.num_warmup_steps,
        num_training_steps=total_steps / training_args.gradient_accumulation_steps
    )

    scaler = GradScaler()

    if training_args.restore_model_path is not None and training_args.resume:
        LOG.info("Restoring optimizer and scheduler")
        optimizer_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "optimizer.pt"),
            map_location="cpu",
        )
        scheduler_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "scheduler.pt"),
            map_location="cpu",
        )
        scaler_checkpoint = torch.load(
            os.path.join(os.path.dirname(training_args.restore_model_path), "scaler.pt"),
            map_location="cpu",
        )
        optimizer.load_state_dict(optimizer_checkpoint)
        scheduler.load_state_dict(scheduler_checkpoint)
        scaler.load_state_dict(scaler_checkpoint)

    run_fine_tuning_loops(
        refined=refined,
        fine_tuning_args=training_args,
        training_dataloader=training_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        evaluation_dataset_name_to_docs={'WIKI_DEV': eval_docs},
        checkpoint_every_n_steps=training_args.checkpoint_every_n_steps
    )

    dq.finish()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()
