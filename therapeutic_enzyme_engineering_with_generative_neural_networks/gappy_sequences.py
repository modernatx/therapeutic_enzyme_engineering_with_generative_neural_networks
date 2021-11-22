"""
Functions for identifying and removing sequences that introduce large gaps in alignment.
This is useful for MSA-based modeling, e.g., for neural networks or conservation analyses.
"""
import pandas as pd
from seqlike.alignment_commands import mafft_alignment
from seqlike.alphabets import gap_letter
from seqlike.encoders import ENCODERS


def gap_score(seqrec, counts, alphabet_letters, motif_depth, threshold=0.95):
    """Score sequences based on gaps introduced to an alignment by the sequence.
    Counts number of gap characters in alignment column for which seqrec has non-gap character,
    i.e., how many gaps introduced by `seqrec`.

    :param seqrec: a SeqRecord
    :param counts: a 2D array (N x L) of letter counts, where N is seqrec length and L is alphabet length
    :param alphabet_letters: list of unique alphabet letters corresponding to dim 2 of `counts`
    :param motif_depth: number of sequences in the alignment
    :param threshold: a gap count threshold (%); will only score column when gap count is
        greater than this percent
    :return: a float representing the gap score
    """
    gap_index = alphabet_letters.index(gap_letter)
    score = 0
    for i, letter in enumerate(seqrec.seq):
        gap_counts = counts[i, gap_index]
        if letter is not gap_letter and gap_counts > threshold * motif_depth:
            score += gap_counts
    return float(score) / motif_depth


def gap_score_sequences(aligned, threshold=0.95):
    """Calculate gap scores for all sequences in an alignment

    :param aligned: a sequence alignment
    :param threshold: a gap count threshold (%); will only score column when gap count is
        greater than this percent
    :return: list of sorted gap scores
    """
    counts = aligned.seq.as_counts(encoder=ENCODERS[aligned.seq._type]["onehot"]["full"])
    gap_scores = [
        (gap_score(seqrec, counts, aligned.seq.alphabet, len(aligned), threshold=threshold), seqrec.id)
        for seqrec in aligned
    ]
    return list(sorted(gap_scores, reverse=True))


def identify_gappy_sequences(aligned, threshold=0.95, min_score=10):
    """Identify sequences within alignment that are responsible for introducing large gaps

    :param aligned: a sequence alignment
    :param threshold: a gap count threshold (%); will only score column when gap count is
        greater than this percent; higher values will yield fewer gappy sequences
    :param min_score: gap score threshold above which a sequence is considered gappy
    :return: seq IDs corresponding to seqs with gap scores greater than min_score.
    """
    gap_scores = gap_score_sequences(aligned, threshold=threshold)
    return [seqid for score, seqid in gap_scores if score > min_score]


def characterize_non_gappy_sequences(aligned, refrec, threshold=0.95, min_score=10, verbose=False):
    """Generate summary dataframe for non-gappy sequences

    :param aligned: a sequence alignment
    :param refrec: a SeqRecord with the reference sequence
    :param threshold: a gap count threshold (%); will only score column when gap count is
        greater than this percent; higher values will yield fewer gappy sequences
    :param min_score: gap score threshold above which a sequence is considered gappy
    :param verbose: if True, print verbose output
    :return: a DataFrame object
    """
    gap_scores = gap_score_sequences(aligned, threshold=threshold)
    ignore = [seqid for score, seqid in gap_scores if score > min_score]
    if verbose:
        print("params: threshold %s, min_score %s" % (threshold, min_score), refrec.id)
        print("ignore:", len(ignore))
    seqrecs = [seqrec for seqrec in aligned if seqrec.id not in ignore]
    sub_aligned = mafft_alignment(seqrecs, preserve_order=False, reorder=False, alphabet=aligned._alphabet)
    alignment_length_mult = sub_aligned.get_alignment_length() / len(refrec)
    if verbose:
        print("non-gappy sequences:", len(seqrecs), len(sub_aligned))
        print("alignment lengths:", aligned.get_alignment_length() / len(refrec), alignment_length_mult)
    return pd.DataFrame(
        [
            {
                "id": seqid,
                "gap score": score,
                "alignment length multiplier": alignment_length_mult,
                "min score": min_score,
                "threshold": threshold,
            }
            for score, seqid in gap_scores
        ]
    )
