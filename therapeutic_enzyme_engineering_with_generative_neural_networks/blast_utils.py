"""Code to search the BLAST databases for a sequence and parse the results"""
import os

from Bio.Blast import NCBIWWW, NCBIXML
from Bio.Blast.Applications import NcbiblastpCommandline
from seqlike import SeqLike
from seqlike.alphabets import gap_letter


def run_protein_blast(seqrec, xmlfile, txids=None, entrez=None, local=False, db="nr", evalue=10, hits=500, **kwargs):
    """Run protein BLAST, return a NCBI records generator

    :param seqrec: TODO
    :param xmlfile: TODO
    :param txids: TODO, defaults to None
    :param entrez: TODO, defaults to None
    :param local: TODO, defaults to False
    :param db: TODO, defaults to "nr"
    :param evalue: TODO, defaults to 10
    :param hits: TODO, defaults to 500
    :param kwargs: TODO
    :return: TODO
    """
    # use cached BLAST XML file if it already exists
    try:
        blast_parser = NCBIXML.parse(open(xmlfile, "r"))
        print("Found cached BLAST XML file %s..." % xmlfile)
    except IOError:
        if txids or entrez:
            entrez_query = list()
            if txids:
                entrez_query.append("(" + " OR ".join("txid%s[ORGN]" % txid for txid in txids) + ")")
            if entrez:
                entrez_query.append(" ".join(entrez))
            entrez_query = " AND ".join(entrez_query)
            print("entrez_query: %s" % entrez_query)
        else:
            entrez_query = None
        if local:
            # outfmt 5 is XML BLAST output [https://www.biostars.org/p/88944/#88949]
            cline = NcbiblastpCommandline(
                db=db,
                entrez_query=entrez_query,
                evalue=evalue,
                ungapped=False,
                out=xmlfile,
                outfmt=5,
                hitlist_size=hits,
                **kwargs
            )
            stdout, stderr = cline(stdin=str(seqrec.seq))
        else:
            result_handle = NCBIWWW.qblast(
                "blastp",
                db,
                seqrec.format("fasta"),
                entrez_query=entrez_query,
                expect=evalue,
                hitlist_size=hits,
                **kwargs
            )
            with open(xmlfile, "w") as f:
                f.write(result_handle.read())
                result_handle.close()
        blast_parser = NCBIXML.parse(open(xmlfile, "r"))
    return blast_parser


def clean_xmlfile(xmlfile, suffix="_clean"):
    """Sometimes the XML file includes garbage that causes the NCBIXML parser to fail

    :param xmlfile: an XML filename
    :param suffix: the suffix to append to the XML filename
    :return: the filename of the cleaned xml file
    """
    xmlfile_out = os.path.splitext(xmlfile)[0] + suffix + ".xml"
    with open(xmlfile_out, "w") as f:
        for line in open(xmlfile, "r"):
            if "CREATE_VIEW" in line:
                pass
            else:
                f.write(line)
    return xmlfile_out


def ungap_to_reference(seq, ref, gap_letter):
    """Assuming aligned seq and ref, slice seq to match ungapped refrec

    :param seq: SeqLikeType of sequence to slice to match reference
    :param ref: SeqLikeType of reference sequence
    :param gap_letter: Symbol to use as a gap letter (e.g. '-')
    :return: ungapped sequence
    """
    col_indices = [i for i, letter in enumerate(ref) if letter != gap_letter]
    return "".join(seq[i] for i in col_indices)


def pad_hsp(seq, start, end, length):
    """Pad sequence

    :param seq: TODO
    :param start: TODO
    :param end: TODO
    :param length: TODO
    :return: TODO
    """
    return gap_letter * (start - 1) + seq + gap_letter * (length - end)


def merge_hsps(hsps, query_length):
    """Helper function for parse_seqrecs_from_blast
    Gaps and overlap often exist, as well as different alignments.
    If overlap conflict exists, return unmerged sequences.

    :param hsps: list of sequence strings
    :param query_length: query sequence
    :return: TODO
    """

    seqs = list()
    for hsp in hsps:
        sbjct = hsp.sbjct.replace("X", gap_letter)
        # BLAST aligns results to query/ref seq; use common ref to pad-align all sequences
        sbjct = ungap_to_reference(sbjct, hsp.query, gap_letter)
        sbjct = pad_hsp(str(sbjct), hsp.query_start, hsp.query_end, query_length)
        seqs.append(sbjct)

    counts = dict()
    for seq in seqs:
        for i, letter in enumerate(seq):
            counts.setdefault(i, list()).append(letter)

    # if any sequence position has more than one non-gap letter, return unmerged seqs
    def count_nongap(position):
        """simple helper function to count all the non-gap characters

        :param position: counts of letters at a given position
        :return: number of non-gap characters
        """
        unique = set(position)
        return len(unique) - (gap_letter in unique)

    if any(count_nongap(count) > 1 for count in counts.values()):
        return seqs
    # otherwise, return the consensus sequence
    else:
        # sorting letters will put non-gap letter at end
        merged = "".join(sorted(set(counts[i]))[-1] for i in range(len(counts)))
        # return a list of seq for consistency
        return [merged]


def parse_seqrecs_from_blast(blastrecs, seq_type="aa", top_hsp_only=False, use_blast_align=True, verbose=False):
    """Generator of SeqRecord based on BLAST records

    :param blastrecs: records from NCBIXML.parse or similar
    :param seq_type: "aa" or "nt" for SeqLike
    :param top_hsp_only: if True, save only top high-scoring pair from each alignment
    :param use_blast_align: prepend/append sequence with gap letters, also slice subject sequences to
        line up with query/ref sequence; resulting seqrecs should be aligned and of equal length
    :param verbose: extra output, defaults to False
    :yield: TODO
    """
    for blastrec in blastrecs:
        for alignment in blastrec.alignments:
            # extract title and description from BLAST query
            alignment_titles = alignment.title.split(">gi")
            title, description = alignment_titles[0].split(" ", 1)
            # iterate through high-scoring pairs
            hsps = sorted(alignment.hsps, key=lambda hsp: hsp.score, reverse=True)
            if verbose:
                for hsp in hsps:
                    print(hsp)
            if top_hsp_only:
                hsps = hsps[0:1]
            # slice and pad subject sequence to match query; merge HSPs when possible
            if use_blast_align:
                seqs = merge_hsps(hsps, blastrec.query_length)
            else:
                seqs = [hsp.sbjct.replace("X", gap_letter) for hsp in hsps]
            # convert the sequences into SeqLike
            for seq in seqs:
                yield SeqLike(
                    seq,
                    seq_type=seq_type,
                    id=title,
                    name=title,
                    description="%s %i seqs" % (description, len(alignment.hsps)),
                )


def parse_genbank_ids_from_blast(blastrecs):
    """Generator of GenBank accession IDs based on BLAST records

    :param blastrecs: records from NCBIXML.parse or similar
    :yield: TODO
    """
    for blastrec in blastrecs:
        for alignment in blastrec.alignments:
            yield alignment.hit_id
