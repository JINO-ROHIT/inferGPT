//check the model_index.json for the mappings

#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "model_quant.h"
#include "tensor.h"

bool load_gpt2_model_quantized(ModelQuant &m) {
  const char *fname = "model/model_absmax.bin";
  int fd = open(fname, O_RDONLY);
  if (fd < 0) {
    perror(fname);
    return false;
  }
  struct stat sb;
  fstat(fd, &sb);
  char *data = (char*) mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);

  m.mmap_data = data;
  m.mmap_siz = sb.st_size;

  m.embedding_dim = 768;
  m.context_len = 1024;
  m.ntokens = 50257;
  m.h = new TransformerBlockQuant[12];
  m.wte_weight                   = Tensor_Quant<2>((int8_t*)(data + 0x00000000), 50257, 768, 71.70240020751953);
  m.wpe_weight                   = Tensor_Quant<2>((int8_t*)(data + 0x024CF300), 1024, 768, 28.20555305480957);
  m.ln_f.bias                    = Tensor_Quant<1>((int8_t*)(data + 0x0258F300), 768, 17.371658325195312);
  m.ln_f.weight                  = Tensor_Quant<1>((int8_t*)(data + 0x0258F600), 768, 7.348164081573486);
  m.h[0].ln_1.bias               = Tensor_Quant<1>((int8_t*)(data + 0x0258F900), 768, 494.4318542480469);
  m.h[0].ln_1.weight             = Tensor_Quant<1>((int8_t*)(data + 0x0258FC00), 768, 506.5961608886719);
  m.h[0].attn.num_heads = 12;
  m.h[0].attn.c_attn_bias        = Tensor_Quant<1>((int8_t*)(data + 0x0258FF00), 2304, 95.73133850097656);
  m.h[0].attn.c_attn_weight      = Tensor_Quant<2>((int8_t*)(data + 0x02590800), 2304, 768, 45.012821197509766);
  m.h[0].attn.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x02740800), 768, 47.6826286315918);
  m.h[0].attn.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x02740B00), 768, 768, 38.587467193603516);
  m.h[0].ln_2.bias               = Tensor_Quant<1>((int8_t*)(data + 0x027D0B00), 768, 173.1176300048828);
  m.h[0].ln_2.weight             = Tensor_Quant<1>((int8_t*)(data + 0x027D0E00), 768, 84.71016693115234);
  m.h[0].mlp.c_fc_bias           = Tensor_Quant<1>((int8_t*)(data + 0x027D1100), 3072, 171.5435028076172);
  m.h[0].mlp.c_fc_weight         = Tensor_Quant<2>((int8_t*)(data + 0x027D1D00), 3072, 768, 27.900571823120117);
  m.h[0].mlp.c_proj_bias         = Tensor_Quant<1>((int8_t*)(data + 0x02A11D00), 768, 86.52379608154297);
  m.h[0].mlp.c_proj_weight       = Tensor_Quant<2>((int8_t*)(data + 0x02A12000), 768, 3072, 20.83587074279785);
  
  m.h[1].ln_1.bias               = Tensor_Quant<1>((int8_t*)(data + 0x02C52000), 768, 192.62098693847656);
  m.h[1].ln_1.weight             = Tensor_Quant<1>((int8_t*)(data + 0x02C52300), 768, 195.32015991210938);
  m.h[1].attn.num_heads = 12;
  m.h[1].attn.c_attn_bias        = Tensor_Quant<1>((int8_t*)(data + 0x02C52600), 2304, 65.99771881103516);
  m.h[1].attn.c_attn_weight      = Tensor_Quant<2>((int8_t*)(data + 0x02C52F00), 2304, 768, 103.36876678466797);
  m.h[1].attn.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x02E02F00), 768, 102.3427505493164);
  m.h[1].attn.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x02E03200), 768, 768, 27.082853317260742);
  m.h[1].ln_2.bias               = Tensor_Quant<1>((int8_t*)(data + 0x02E93200), 768, 218.20272827148438);
  m.h[1].ln_2.weight             = Tensor_Quant<1>((int8_t*)(data + 0x02E93500), 768, 283.0008544921875);
  m.h[1].mlp.c_fc_bias           = Tensor_Quant<1>((int8_t*)(data + 0x02E93800), 3072, 195.03785705566406);
  m.h[1].mlp.c_fc_weight         = Tensor_Quant<2>((int8_t*)(data + 0x02E94400), 3072, 768, 55.91432571411133);
  m.h[1].mlp.c_proj_bias         = Tensor_Quant<1>((int8_t*)(data + 0x030D4400), 768, 80.28211975097656);
  m.h[1].mlp.c_proj_weight       = Tensor_Quant<2>((int8_t*)(data + 0x030D4700), 768, 3072, 9.31843090057373);
  
  m.h[2].ln_1.bias               = Tensor_Quant<1>((int8_t*)(data + 0x03314700), 768, 117.0459213256836);
  m.h[2].ln_1.weight             = Tensor_Quant<1>((int8_t*)(data + 0x03314A00), 768, 135.54698181152344);
  m.h[2].attn.num_heads = 12;
  m.h[2].attn.c_attn_bias        = Tensor_Quant<1>((int8_t*)(data + 0x03314D00), 2304, 87.4916000366211);
  m.h[2].attn.c_attn_weight      = Tensor_Quant<2>((int8_t*)(data + 0x03315600), 2304, 768, 76.30789947509766);
  m.h[2].attn.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x034C5600), 768, 248.6093292236328);
  m.h[2].attn.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x034C5900), 768, 768, 55.7601318359375);
  m.h[2].ln_2.bias               = Tensor_Quant<1>((int8_t*)(data + 0x03555900), 768, 197.84024047851562);
  m.h[2].ln_2.weight             = Tensor_Quant<1>((int8_t*)(data + 0x03555C00), 768, 175.4354248046875);
  m.h[2].mlp.c_fc_bias           = Tensor_Quant<1>((int8_t*)(data + 0x03555F00), 3072, 73.92515563964844);
  m.h[2].mlp.c_fc_weight         = Tensor_Quant<2>((int8_t*)(data + 0x03556B00), 3072, 768, 12.126031875610352);
  m.h[2].mlp.c_proj_bias         = Tensor_Quant<1>((int8_t*)(data + 0x03796B00), 768, 81.65037536621094);
  m.h[2].mlp.c_proj_weight       = Tensor_Quant<2>((int8_t*)(data + 0x03796E00), 768, 3072, 8.494786262512207);
  
  m.h[3].ln_1.bias               = Tensor_Quant<1>((int8_t*)(data + 0x039D6E00), 768, 73.67604064941406);
  m.h[3].ln_1.weight             = Tensor_Quant<1>((int8_t*)(data + 0x039D7100), 768, 166.73831176757812);
  m.h[3].attn.num_heads = 12;
  m.h[3].attn.c_attn_bias        = Tensor_Quant<1>((int8_t*)(data + 0x039D7400), 2304, 179.8214874267578);
  m.h[3].attn.c_attn_weight      = Tensor_Quant<2>((int8_t*)(data + 0x039D7D00), 2304, 768, 67.51042938232422);
  m.h[3].attn.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x03B87D00), 768, 124.6077880859375);
  m.h[3].attn.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x03B88000), 768, 768, 61.1342658996582);
  m.h[3].ln_2.bias               = Tensor_Quant<1>((int8_t*)(data + 0x03C18000), 768, 292.58770751953125);
  m.h[3].ln_2.weight             = Tensor_Quant<1>((int8_t*)(data + 0x03C18300), 768, 110.30354309082031);
  m.h[3].mlp.c_fc_bias           = Tensor_Quant<1>((int8_t*)(data + 0x03C18600), 3072, 104.38511657714844);
  m.h[3].mlp.c_fc_weight         = Tensor_Quant<2>((int8_t*)(data + 0x03C19200), 3072, 768, 48.416133880615234);
  m.h[3].mlp.c_proj_bias         = Tensor_Quant<1>((int8_t*)(data + 0x03E59200), 768, 68.67874145507812);
  m.h[3].mlp.c_proj_weight       = Tensor_Quant<2>((int8_t*)(data + 0x03E59500), 768, 3072, 7.484360694885254);
  
  m.h[4].ln_1.bias               = Tensor_Quant<1>((int8_t*)(data + 0x04099500), 768, 82.53905487060547);
  m.h[4].ln_1.weight             = Tensor_Quant<1>((int8_t*)(data + 0x04099800), 768, 190.91322326660156);
  m.h[4].attn.num_heads = 12;
  m.h[4].attn.c_attn_bias        = Tensor_Quant<1>((int8_t*)(data + 0x04099B00), 2304, 46.6187629699707);
  m.h[4].attn.c_attn_weight      = Tensor_Quant<2>((int8_t*)(data + 0x0409A400), 2304, 768, 38.391666412353516);
  m.h[4].attn.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x0424A400), 768, 199.06556701660156);
  m.h[4].attn.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x0424A700), 768, 768, 69.61231994628906);
  m.h[4].ln_2.bias               = Tensor_Quant<1>((int8_t*)(data + 0x042DA700), 768, 895.4752197265625);
  m.h[4].ln_2.weight             = Tensor_Quant<1>((int8_t*)(data + 0x042DAA00), 768, 113.05570983886719);
  m.h[4].mlp.c_fc_bias           = Tensor_Quant<1>((int8_t*)(data + 0x042DAD00), 3072, 172.5740966796875);
  m.h[4].mlp.c_fc_weight         = Tensor_Quant<2>((int8_t*)(data + 0x042DB900), 3072, 768, 59.221839904785156);
  m.h[4].mlp.c_proj_bias         = Tensor_Quant<1>((int8_t*)(data + 0x0451B900), 768, 81.60187530517578);
  m.h[4].mlp.c_proj_weight       = Tensor_Quant<2>((int8_t*)(data + 0x0451BC00), 768, 3072, 26.794424057006836);
  
  m.h[5].ln_1.bias               = Tensor_Quant<1>((int8_t*)(data + 0x0475BC00), 768, 118.57815551757812);
  m.h[5].ln_1.weight             = Tensor_Quant<1>((int8_t*)(data + 0x0475BF00), 768, 166.51416015625);
  m.h[5].attn.num_heads = 12;
  m.h[5].attn.c_attn_bias        = Tensor_Quant<1>((int8_t*)(data + 0x0475C200), 2304, 228.8199920654297);
  m.h[5].attn.c_attn_weight      = Tensor_Quant<2>((int8_t*)(data + 0x0475CB00), 2304, 768, 91.9284896850586);
  m.h[5].attn.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x0490CB00), 768, 182.08070373535156);
  m.h[5].attn.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x0490CE00), 768, 768, 64.53772735595703);
  m.h[5].ln_2.bias               = Tensor_Quant<1>((int8_t*)(data + 0x0499CE00), 768, 402.7437438964844);
  m.h[5].ln_2.weight             = Tensor_Quant<1>((int8_t*)(data + 0x0499D100), 768, 90.27000427246094);
  m.h[5].mlp.c_fc_bias           = Tensor_Quant<1>((int8_t*)(data + 0x0499D400), 3072, 191.01675415039062);
  m.h[5].mlp.c_fc_weight         = Tensor_Quant<2>((int8_t*)(data + 0x0499E000), 3072, 768, 65.15449523925781);
  m.h[5].mlp.c_proj_bias         = Tensor_Quant<1>((int8_t*)(data + 0x04BDE000), 768, 102.18983459472656);
  m.h[5].mlp.c_proj_weight       = Tensor_Quant<2>((int8_t*)(data + 0x04BDE300), 768, 3072, 45.059757232666016);
  
  m.h[6].ln_1.bias               = Tensor_Quant<1>((int8_t*)(data + 0x04E1E300), 768, 84.12530517578125);
  m.h[6].ln_1.weight             = Tensor_Quant<1>((int8_t*)(data + 0x04E1E600), 768, 164.1765594482422);
  m.h[6].attn.num_heads = 12;
  m.h[6].attn.c_attn_bias        = Tensor_Quant<1>((int8_t*)(data + 0x04E1E900), 2304, 157.7453155517578);
  m.h[6].attn.c_attn_weight      = Tensor_Quant<2>((int8_t*)(data + 0x04E1F200), 2304, 768, 78.71436309814453);
  m.h[6].attn.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x04FCF200), 768, 323.2585144042969);
  m.h[6].attn.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x04FCF500), 768, 768, 68.87967681884766);
  m.h[6].ln_2.bias               = Tensor_Quant<1>((int8_t*)(data + 0x0505F500), 768, 283.1087646484375);
  m.h[6].ln_2.weight             = Tensor_Quant<1>((int8_t*)(data + 0x0505F800), 768, 95.5335693359375);
  m.h[6].mlp.c_fc_bias           = Tensor_Quant<1>((int8_t*)(data + 0x0505FB00), 3072, 187.5805206298828);
  m.h[6].mlp.c_fc_weight         = Tensor_Quant<2>((int8_t*)(data + 0x05060700), 3072, 768, 60.01472473144531);
  m.h[6].mlp.c_proj_bias         = Tensor_Quant<1>((int8_t*)(data + 0x052A0700), 768, 122.37699127197266);
  m.h[6].mlp.c_proj_weight       = Tensor_Quant<2>((int8_t*)(data + 0x052A0A00), 768, 3072, 46.210289001464844);
  
  m.h[7].ln_1.bias               = Tensor_Quant<1>((int8_t*)(data + 0x054E0A00), 768, 106.98958587646484);
  m.h[7].ln_1.weight             = Tensor_Quant<1>((int8_t*)(data + 0x054E0D00), 768, 156.43063354492188);
  m.h[7].attn.num_heads = 12;
  m.h[7].attn.c_attn_bias        = Tensor_Quant<1>((int8_t*)(data + 0x054E1000), 2304, 169.145263671875);
  m.h[7].attn.c_attn_weight      = Tensor_Quant<2>((int8_t*)(data + 0x054E1900), 2304, 768, 70.90717315673828);
  m.h[7].attn.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x05691900), 768, 247.04074096679688);
  m.h[7].attn.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x05691C00), 768, 768, 57.28632736206055);
  m.h[7].ln_2.bias               = Tensor_Quant<1>((int8_t*)(data + 0x05721C00), 768, 204.08058166503906);
  m.h[7].ln_2.weight             = Tensor_Quant<1>((int8_t*)(data + 0x05721F00), 768, 98.99679565429688);
  m.h[7].mlp.c_fc_bias           = Tensor_Quant<1>((int8_t*)(data + 0x05722200), 3072, 153.84481811523438);
  m.h[7].mlp.c_fc_weight         = Tensor_Quant<2>((int8_t*)(data + 0x05722E00), 3072, 768, 103.8806381225586);
  m.h[7].mlp.c_proj_bias         = Tensor_Quant<1>((int8_t*)(data + 0x05962E00), 768, 108.75040435791016);
  m.h[7].mlp.c_proj_weight       = Tensor_Quant<2>((int8_t*)(data + 0x05963100), 768, 3072, 28.238813400268555);
  
  m.h[8].ln_1.bias               = Tensor_Quant<1>((int8_t*)(data + 0x05BA3100), 768, 88.03301239013672);
  m.h[8].ln_1.weight             = Tensor_Quant<1>((int8_t*)(data + 0x05BA3400), 768, 138.41046142578125);
  m.h[8].attn.num_heads = 12;
  m.h[8].attn.c_attn_bias        = Tensor_Quant<1>((int8_t*)(data + 0x05BA3700), 2304, 147.28939819335938);
  m.h[8].attn.c_attn_weight      = Tensor_Quant<2>((int8_t*)(data + 0x05BA4000), 2304, 768, 65.70748901367188);
  m.h[8].attn.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x05D54000), 768, 110.59455108642578);
  m.h[8].attn.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x05D54300), 768, 768, 43.53959655761719);
  m.h[8].ln_2.bias               = Tensor_Quant<1>((int8_t*)(data + 0x05DE4300), 768, 184.1848907470703);
  m.h[8].ln_2.weight             = Tensor_Quant<1>((int8_t*)(data + 0x05DE4600), 768, 119.38114166259766);
  m.h[8].mlp.c_fc_bias           = Tensor_Quant<1>((int8_t*)(data + 0x05DE4900), 3072, 128.88107299804688);
  m.h[8].mlp.c_fc_weight         = Tensor_Quant<2>((int8_t*)(data + 0x05DE5500), 3072, 768, 88.0060043334961);
  m.h[8].mlp.c_proj_bias         = Tensor_Quant<1>((int8_t*)(data + 0x06025500), 768, 104.89472961425781);
  m.h[8].mlp.c_proj_weight       = Tensor_Quant<2>((int8_t*)(data + 0x06025800), 768, 3072, 23.818347930908203);
  
  m.h[9].ln_1.bias               = Tensor_Quant<1>((int8_t*)(data + 0x06265800), 768, 101.26683044433594);
  m.h[9].ln_1.weight             = Tensor_Quant<1>((int8_t*)(data + 0x06265B00), 768, 135.4503936767578);
  m.h[9].attn.num_heads = 12;
  m.h[9].attn.c_attn_bias        = Tensor_Quant<1>((int8_t*)(data + 0x06265E00), 2304, 122.78771209716797);
  m.h[9].attn.c_attn_weight      = Tensor_Quant<2>((int8_t*)(data + 0x06266700), 2304, 768, 64.45108032226562);
  m.h[9].attn.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x06416700), 768, 67.51008605957031);
  m.h[9].attn.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x06416A00), 768, 768, 64.42272186279297);
  m.h[9].ln_2.bias               = Tensor_Quant<1>((int8_t*)(data + 0x064A6A00), 768, 228.16079711914062);
  m.h[9].ln_2.weight             = Tensor_Quant<1>((int8_t*)(data + 0x064A6D00), 768, 135.07901000976562);
  m.h[9].mlp.c_fc_bias           = Tensor_Quant<1>((int8_t*)(data + 0x064A7000), 3072, 209.73623657226562);
  m.h[9].mlp.c_fc_weight         = Tensor_Quant<2>((int8_t*)(data + 0x064A7C00), 3072, 768, 46.013755798339844);
  m.h[9].mlp.c_proj_bias         = Tensor_Quant<1>((int8_t*)(data + 0x066E7C00), 768, 85.92892456054688);
  m.h[9].mlp.c_proj_weight       = Tensor_Quant<2>((int8_t*)(data + 0x066E7F00), 768, 3072, 23.325851440429688);
  
  m.h[10].ln_1.bias              = Tensor_Quant<1>((int8_t*)(data + 0x06927F00), 768, 117.48451232910156);
  m.h[10].ln_1.weight            = Tensor_Quant<1>((int8_t*)(data + 0x06928200), 768, 138.92327880859375);
  m.h[10].attn.num_heads = 12;
  m.h[10].attn.c_attn_bias       = Tensor_Quant<1>((int8_t*)(data + 0x06928500), 2304, 139.7032470703125);
  m.h[10].attn.c_attn_weight     = Tensor_Quant<2>((int8_t*)(data + 0x06928E00), 2304, 768, 67.58914184570312);
  m.h[10].attn.c_proj_bias       = Tensor_Quant<1>((int8_t*)(data + 0x06AD8E00), 768, 33.25480270385742);
  m.h[10].attn.c_proj_weight     = Tensor_Quant<2>((int8_t*)(data + 0x06AD9100), 768, 768, 30.388160705566406);
  m.h[10].ln_2.bias              = Tensor_Quant<1>((int8_t*)(data + 0x06B69100), 768, 185.14187622070312);
  m.h[10].ln_2.weight            = Tensor_Quant<1>((int8_t*)(data + 0x06B69400), 768, 116.888916015625);
  m.h[10].mlp.c_fc_bias          = Tensor_Quant<1>((int8_t*)(data + 0x06B69700), 3072, 119.31031036376953);
  m.h[10].mlp.c_fc_weight        = Tensor_Quant<2>((int8_t*)(data + 0x06B6A300), 3072, 768, 50.126407623291016);
  m.h[10].mlp.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x06DAA300), 768, 95.49342346191406);
  m.h[10].mlp.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x06DAA600), 768, 3072, 11.583242416381836);
  
  m.h[11].ln_1.bias              = Tensor_Quant<1>((int8_t*)(data + 0x06FEA600), 768, 127.50188446044922);
  m.h[11].ln_1.weight            = Tensor_Quant<1>((int8_t*)(data + 0x06FEA900), 768, 133.6591339111328);
  m.h[11].attn.num_heads = 12;
  m.h[11].attn.c_attn_bias       = Tensor_Quant<1>((int8_t*)(data + 0x06FEAC00), 2304, 159.08602905273438);
  m.h[11].attn.c_attn_weight     = Tensor_Quant<2>((int8_t*)(data + 0x06FEB500), 2304, 768, 55.47988510131836);
  m.h[11].attn.c_proj_bias       = Tensor_Quant<1>((int8_t*)(data + 0x0719B500), 768, 23.823333740234375);
  m.h[11].attn.c_proj_weight     = Tensor_Quant<2>((int8_t*)(data + 0x0719B800), 768, 768, 14.41146469116211);
  m.h[11].ln_2.bias              = Tensor_Quant<1>((int8_t*)(data + 0x0722B800), 768, 301.4186096191406);
  m.h[11].ln_2.weight            = Tensor_Quant<1>((int8_t*)(data + 0x0722BB00), 768, 103.81061553955078);
  m.h[11].mlp.c_fc_bias          = Tensor_Quant<1>((int8_t*)(data + 0x0722BE00), 3072, 105.94122314453125);
  m.h[11].mlp.c_fc_weight        = Tensor_Quant<2>((int8_t*)(data + 0x0722CA00), 3072, 768, 65.14434814453125);
  m.h[11].mlp.c_proj_bias        = Tensor_Quant<1>((int8_t*)(data + 0x0746CA00), 768, 292.67333984375);
  m.h[11].mlp.c_proj_weight      = Tensor_Quant<2>((int8_t*)(data + 0x0746CD00), 768, 3072, 13.895626068115234);

  close(fd);
  return true;
}