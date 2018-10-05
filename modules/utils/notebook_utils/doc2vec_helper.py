import subprocess

from snorkel.models import Candidate

def get_candidate_objects(session, candidate_dicts):
    """
    This function is designed to get candiadte objects from the postgres database

    session - an open connection to the postgres database
    candidate_dicts - a dictionary containing the candidate dataframes
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
    This function is designed to get candiadte objects from the postgres database

    candidate_obj_dict - a dictionary containing candidates objects
    sentences_file_dict - a dictionary containing directories to write each file to.
    """

    for key in candidate_obj_dict:
        with open(sentences_file_dict[key], 'w') as f:
            for candidate in candidate_obj_dict[key]:
                f.write(candidate.get_parent().text + "\n")

    return 

def execute_doc2vec(training_file, word_file, output_file, test_file, vocab_file, read_vocab=False):
    """
    This function is designed run doc2vec to embed sentences into dense vectors.

    training_file - a file path that contains training sentences used for embeddings
    word_file - a file path that tells doc2vec where to output word vectors
    output_file - a file path that tells doc2vec where to output doc vectors
    test_file - a file path that contains sentences that will be embedded
    vocab_file - a file path that tells doc2vec where to output vocabulary
    read_vocab - a boolean to say read in a vocab list or create one from scratch
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
