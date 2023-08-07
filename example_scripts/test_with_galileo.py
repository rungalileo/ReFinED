import sys
import os
import numpy as np

sys.path.insert(0, "src")

from refined.dataset_reading.entity_linking.dataset_factory import Datasets
from refined.training.fine_tune.fine_tune_args import parse_fine_tuning_args, FineTuningArgs
from refined.inference.processor import Refined
from refined.utilities.galileo_helpers import log_input_data_galileo
from refined.evaluation.evaluation import evaluate, get_datasets_obj

import dataquality as dq

USE_ALL_ENTITY_TYPES = True
GALILEO_PROJECT_NAME = "MusixMatchFull"
BATCH_SIZE = 100
PODCAST_ENG_SPLIT = "dev"


def run_batch_inference_job(
        refined: Refined,
        datasets: Datasets,
        fine_tuning_args: FineTuningArgs,
        run_name: str
) -> None:
    dq.init(task_type="text_multi_label",
            project_name=GALILEO_PROJECT_NAME,
            run_name=run_name)

    # Load the fine-tuning dataset.
    # NOTE: we load this for each batch because of strange issues
    # where refined overwrites data during the eval procedure making it
    # hard to run multiple times over the dataset.
    evaluation_dataset_name_to_docs = {
        "PODCAST-ENG": list(datasets.get_podcast_docs(
            # TODO Specify split
            split=PODCAST_ENG_SPLIT
        ))
    }
    print("Number of documents:", len(evaluation_dataset_name_to_docs['AIDA']))

    # 🔭🌕 Galileo logging
    log_input_data_galileo(evaluation_dataset_name_to_docs["AIDA"], refined.preprocessor, "test", 8)

    # Run the model over the documents using the evaluate function
    dq.set_split("test")
    dq.set_epoch(0)
    evaluate(
        refined=refined,
        evaluation_dataset_name_to_docs=evaluation_dataset_name_to_docs,
        el=False,  # No need to eval EL
        ed=True,
        ed_threshold=fine_tuning_args.ed_threshold
    )

    dq.finish()


def main():
    # 🔭🌕 Galileo logging
    # Initialize Project
    dq.configure()

    # Load the model and fine-tuning dataset
    fine_tuning_args = parse_fine_tuning_args()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    refined = Refined.from_pretrained(model_name=fine_tuning_args.model_name,
                                      entity_set=fine_tuning_args.entity_set,
                                      use_precomputed_descriptions=fine_tuning_args.use_precomputed_descriptions,
                                      device=fine_tuning_args.device)

    datasets = get_datasets_obj(preprocessor=refined.preprocessor,
                                download=False,
                                data_dir=fine_tuning_args.data_dir)

    # 🔭🌕 Galileo Extract the entity ids that we are logging with Galileo.
    entity_type_ids = refined.preprocessor.lookups.label_subset_arr
    if USE_ALL_ENTITY_TYPES:
        # Create the list of all the entity type ids
        entity_type_ids = np.sort(list(refined.preprocessor.index_to_class.keys()))

    # 🔭🌕 Galileo - Batch processing of the entity-types.
    for i, batch_start_idx in enumerate(range(0, entity_type_ids.shape[0], BATCH_SIZE)):
        print(
            f"Starting Batch {i} with entity id range {entity_type_ids[batch_start_idx]} "
            f"- {entity_type_ids[batch_start_idx + BATCH_SIZE - 1]}"
        )
        entity_types_subset = entity_type_ids[batch_start_idx: batch_start_idx + BATCH_SIZE]

        # Track just the entity types in the current batch
        refined.preprocessor.lookups.label_subset_arr = entity_types_subset
        refined.preprocessor.lookups.label_subset_set = set(entity_types_subset)

        # Launch a Galileo Test Job
        run_name = f"Batch_Run_{i}"
        run_batch_inference_job(
            refined,
            datasets,
            fine_tuning_args,
            run_name
        )


if __name__ == '__main__':
    main()
