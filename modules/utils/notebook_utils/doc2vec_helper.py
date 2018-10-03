import subprocess

from sklearn.model_selection import GridSearchCV
from snorkel.models import Candidate

def get_candidate_objects(session, candidate_dicts):
    """
    """

    return {
    key:(
        session
        .query(Candidate)
        .filter(Candidate.id.in_(candidate_dicts[key].candidate_id.astype(int).tolist()))
        .all()
    )
    for key in candidate_dicts
    }


def write_sentences_to_file(candidate_obj_dict, sentences_file_dict):
    """
    """

    for key in candidate_obj_dict:
        with open(sentences_file_dict[key], 'w') as f:
            for candidate in candidate_obj_dict[key]:
                f.write(candidate.get_parent().text + "\n")

    return 

def execute_doc2vec(training_file, word_file, output_file, test_file, vocab_file, read_vocab=False):
    """
    """

    command = [
    '../../../iclr2017/doc2vecc',
    '-train', training_file,
    '-word', word_file,
    '-output', output_file,
    '-cbow', '1',
    '-size', '500',
    '-negative', '5',
    '-hs', '0',
    '-threads', '5',
    '-binary', '0',
    '-iter', '30',
    '-test', test_file,
    ]

    if read_vocab:
        command + ['read-vocab', vocab_file]
    else:
        command  + ['save-vocab', vocab_file]

    subprocess.Popen(command).wait()
    return

def run_grid_search(model, data,  grid, labels):
    """
    """

    searcher = GridSearchCV(model, param_grid=grid, cv=10, return_train_score=True)
    return searcher.fit(data, labels)

