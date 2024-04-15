
#include <malloc.h>
#include <stdlib.h>

#include "../lib/mcraw.h"

// Read a mcraw file
// Export audio to a PCM data file

//-----------------------------------------------------------------------------
int main(int argc, const char * argv[])
{
    mr_ctx_t *ctx;

    if (argc < 2)
    {
        printf("Not enough parameters!\n Usage: %s filename.mcraw audio_filename", argv[0]);
        exit(0);
    }

    const char *mcraw_filename  = argv[1];
    const char *audio_output_fn = argv[2];
    ctx = mr_decoder_new(0);

    if (ctx == NULL) {
        return -1;
    }

    int res = mr_decoder_open(ctx, mcraw_filename);
    if (res != 0) {
        goto done;
    }

    res = mr_decoder_parse(ctx);
    if (res != 0) {
        goto done;
    }

    {
        FILE *fd = mr_get_file_handle(ctx);

        uint32_t nb_audio_packets = mr_get_audio_packet_count(ctx);
        mr_buffer_offset_t *audio_index = mr_get_audio_index(ctx);

        mr_packet_t pkt = {};

        FILE *fd_audio = fopen(audio_output_fn, "wb");

        for (int i = 0; i < nb_audio_packets; i++)
        {
            res = mr_read_audio_packet(fd, audio_index[i].offset, &pkt);

            if (res != 0) {
                break;
            }

            fwrite(pkt.data, pkt.size, 1, fd_audio);
        }

        fclose(fd_audio);
        mr_packet_free(&pkt);
    }

done:
    mr_decoder_close(ctx);
    mr_decoder_free(ctx);

    return 0;
}
