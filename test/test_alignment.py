from spell_correct.utils import AlignmentHandler

alignment_handler = AlignmentHandler(already_split=False)
src = ['حل عني بقا كأير روح حيك ب غير مسله منيش جاي اتسلا معك ف متفكرني عمب امزح']
tgt = ['حل عني بقى كأير روح حيك بغير مسله منيش جايي اتسلى معك فما تفكرني عم بمزح']
src, tgt = alignment_handler.merge_split_src_tgt(
    src, tgt)
pass
