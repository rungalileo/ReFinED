from typing import List

from refined.data_types.doc_types import Doc
from refined.data_types.modelling_types import BatchedElementsTns
from refined.dataset_reading.entity_linking.wikipedia_dataset import WikipediaDataset
from refined.doc_preprocessing.preprocessor import PreprocessorInferenceOnly
from refined.utilities.preprocessing_utils import convert_doc_to_tensors

from tqdm import tqdm
import dataquality as dq


# 🔭🌕 Galileo
def get_span_context(
    text: str, span_start: int, span_len: int
) -> str:
    """Generate a span's text representation for logging with Galileo

    Returns the full sentence that the span appears in with
     the span highlighted as <<span>>.

    Example:
        <<England>> won the world cup in 20xx.
    """
    pre_span = text[:span_start]
    # Simple search for canonical punctuation
    sentance_start = max(
        pre_span.rfind("."),
        pre_span.rfind("?"),
        pre_span.rfind("!"),
        pre_span.rfind("\n"),
    ) + 1

    sentance_end = len(text) - 1
    for punc in ["?", "!", ".", "\n"]:
        end_punctuation = text.find(punc, span_start + span_len)
        if end_punctuation != -1:
            sentance_end = min(sentance_end, end_punctuation)

    # Include the punctuation itself!
    sentance_end += 1

    span_text_with_context = f"{text[sentance_start:span_start]}" \
                             f"<<{text[span_start: span_start+span_len]}>>" \
                             f"{text[span_start+span_len: sentance_end]}"
    return span_text_with_context.strip()


# 🔭🌕 Galileo
def log_input_data_galileo(
    dataset: WikipediaDataset,
    preprocessor: PreprocessorInferenceOnly,
    split: str,
    max_batch_size: int,
) -> None:
    """Log entity span data with Galileo as MLTC data"""
    lookups = preprocessor.lookups
    # Create map from idx --> entity type name. Used for logging readable labels
    # Shift every by one to allow for the dummy 0th class
    idx_to_label_name = {idx: lookups.class_to_label[lookups.index_to_class[idx]] for idx in lookups.index_to_class.keys()}

    # 🔭🌕 Galileo logging
    # Convert the tasks to their string representations
    task_ids = lookups.label_subset_arr
    tasks = [str(idx_to_label_name[idx]) for idx in list(task_ids)]
    dq.set_tasks_for_run(tasks, binary=True)

    # Loop over the BatchElementTns
    span_texts = []
    spans = []
    span_labels = []
    span_ids = []
    meta_data = {"doc_id": [], "is_md_span": [], "entity_id": []}

    # To ensure sequential reading of the full dataset, set num_workers to 1
    num_workers = dataset.num_workers
    dataset.num_workers = 1
    for data in tqdm(dataset):
        # Handle the Validation dataset case where we have must convert documents first
        # to a list of BatchedElementsTns
        batches: List[BatchedElementsTns] = []
        if type(data) == Doc:
            # type(batch) == DOC -> convert it to a list of BatchedElementsTns.
            for batch_doc_tns in convert_doc_to_tensors(
                data,
                preprocessor,
                collate=True,
                max_batch_size=max_batch_size,
                sort_by_tokens=False,
                max_seq=preprocessor.max_seq,
            ):
                batches.append(batch_doc_tns)
        else:
            batches = [data]

        for batch in batches:
            for i, batch_element in enumerate(batch.batch_elements):
                for j, span in enumerate(batch_element.spans):
                    labels = batch.class_target_values[i][j]
                    labels = labels[labels != 0].numpy()
                    labels = [str(idx_to_label_name[idx]) for idx in labels if idx in preprocessor.lookups.label_subset_set]
                    if len(labels) == 0:
                        continue

                    span_text = get_span_context(batch_element.text, span.start, span.ln)

                    span_texts.append(span_text)
                    spans.append(span.text)
                    span_labels.append(labels)
                    span_ids.append(batch_element.span_ids[j])

                    meta_data['doc_id'].append(span.doc_id)
                    meta_data['is_md_span'].append(span.is_md_span)
                    entity_id = span.gold_entity.wikidata_entity_id or "Q-None"
                    meta_data['entity_id'].append(entity_id)

    # Reset num workers
    dataset.num_workers = num_workers
    # 🔭🌕 Galileo logging
    dq.log_data_samples(
        texts=span_texts,
        task_labels=span_labels,
        ids=span_ids,
        split=split,
        meta=meta_data
    )