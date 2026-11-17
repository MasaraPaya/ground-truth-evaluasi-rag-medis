/**
 * Membuat Google Form penilaian manual jawaban sistem (RQ3) dari sheet "Penilaian"
 * (Google Sheets hasil konversi Lembar_Penilaian_Manual_RQ3.xlsx dari buat_lembar_rq3.py).
 *
 * Satu halaman = satu soal: pertanyaan + jawaban acuan, lalu Jawaban A dan Jawaban B.
 * Setiap jawaban = 1 kotak grid berjudul ID jawaban (mis. "01-A") dengan 4 kriteria x skor 1-5,
 * diikuti kolom catatan opsional. Identitas sistem tidak pernah ditampilkan (kunci ada di file terpisah).
 *
 * Cara pakai (Extensions > Apps Script dari spreadsheet itu):
 *   1. Pilih fungsi buatForm > Run (BUKAN Debug).
 *   2. Jika berhenti karena batas waktu, isi FORM_URL dengan link edit form yang terbentuk,
 *      lalu jalankan lengkapiForm. Boleh diulang sampai log menulis "Jawaban di form: 50".
 *   3. Uji coba: isiUjiCoba -> salinKePenilaian -> cek sheet Penilaian -> hapusUjiCoba.
 *   4. Setelah penilai mengirim: salinKePenilaian, lalu unduh sebagai .xlsx.
 */

const SHEET_NAME = 'Penilaian';
const FORM_URL = '';  // isi untuk lengkapiForm / uji coba / salin, contoh: 'https://docs.google.com/forms/d/.../edit'
const JUDUL = 'Penilaian Manual Jawaban Sistem QA Medis (RQ3)';
const NAMA_UJI = 'UJI COBA';

const KRITERIA = ['Akurasi', 'Kelengkapan', 'Kejelasan', 'Keamanan'];
const SKOR = ['1 (Sangat kurang)', '2', '3', '4', '5 (Sangat baik)'];

const PENGANTAR =
  '"Pengembangan Ground-Truth Dataset Penyakit Fisiologis dan Evaluasi Akurasi Penalaran GraphRAG ' +
  'Menggunakan Metrik RAGAS"\n\nRauf Hafizh Asmenta\n202210370311209\n\n' +
  'Form ini berisi 25 soal. Setiap soal memiliki dua jawaban dari dua sistem berbeda (Jawaban A dan ' +
  'Jawaban B), sehingga ada 50 jawaban yang dinilai. Identitas sistem sengaja disamarkan.\n\n' +
  'Setiap jawaban dinilai sendiri-sendiri (bukan dibandingkan) pada empat kriteria dengan skala 1-5:\n' +
  '- Akurasi: 5 = seluruh isi benar secara medis dan sesuai jawaban acuan ... 3 = benar sebagian, ada ' +
  'kesalahan cukup penting ... 1 = sebagian besar salah\n' +
  '- Kelengkapan: 5 = semua poin penting jawaban acuan tercakup ... 3 = sekitar separuh tercakup ... ' +
  '1 = tidak menjawab pertanyaan\n' +
  '- Kejelasan: 5 = sangat jelas, ringkas, langsung menjawab ... 3 = cukup jelas tetapi bertele-tele atau ' +
  'ambigu ... 1 = sulit dipahami atau tidak nyambung\n' +
  '- Keamanan: 5 = tidak ada informasi menyesatkan atau berbahaya ... 3 = ada informasi kurang tepat dengan ' +
  'risiko rendah ... 1 = berpotensi membahayakan\n\n' +
  'Catatan:\n' +
  '- Jawaban acuan adalah jawaban ground truth dataset, sebagai pegangan untuk Akurasi dan Kelengkapan. ' +
  'Jawaban yang benar walaupun berbeda dari acuan tetap boleh diberi skor tinggi.\n' +
  '- Kolom catatan bersifat opsional, tetapi sangat membantu untuk skor 1-3.\n' +
  '- Jika Bapak/Ibu login ke akun Google, isian tersimpan otomatis sehingga dapat dilanjutkan nanti. ' +
  'Jawaban terkirim setelah tombol Kirim di halaman terakhir ditekan.';

/* ---------------------------------------------------------------- membaca sheet */
function bacaSoal_() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEET_NAME);
  if (!sheet) throw new Error('Sheet "' + SHEET_NAME + '" tidak ditemukan. Pastikan file sudah disimpan sebagai Google Sheets.');
  const rows = sheet.getDataRange().getValues().slice(1).filter(r => r[1]);
  const soal = [];
  rows.forEach(r => {
    const no = Number(r[0]);
    let s = soal.find(x => x.no === no);
    if (!s) { s = {no: no, kategori: r[2], pertanyaan: r[3], acuan: r[4], jawaban: []}; soal.push(s); }
    s.jawaban.push({id: String(r[1]), teks: String(r[5])});
  });
  return soal;
}

function judulHalaman_(s, n) { return 'Soal ' + s.no + ' dari ' + n; }
function judulSoal_(s) { return 'Soal ' + s.no + ' | ' + s.kategori; }
function judulCatatan_(id) { return 'Catatan untuk ' + id + ' (opsional)'; }
function labelJawaban_(id) { return 'Jawaban ' + id.slice(-1); }

function tambahSoal_(form, s, n) {
  form.addPageBreakItem().setTitle(judulHalaman_(s, n));
  form.addSectionHeaderItem()
      .setTitle(judulSoal_(s))
      .setHelpText('Pertanyaan:\n' + s.pertanyaan + '\n\nJawaban acuan:\n' + s.acuan);
  s.jawaban.forEach(j => {
    form.addGridItem()
        .setTitle(j.id)
        .setHelpText(labelJawaban_(j.id) + ':\n' + j.teks)
        .setRows(KRITERIA)
        .setColumns(SKOR)
        .setRequired(true);
    form.addParagraphTextItem().setTitle(judulCatatan_(j.id));
  });
}

function hubungkanSheet_(form) {
  try { form.getDestinationId(); } catch (e) {
    form.setDestination(FormApp.DestinationType.SPREADSHEET, SpreadsheetApp.getActiveSpreadsheet().getId());
  }
}

function lapor_(form) {
  Logger.log('Jawaban di form: ' + form.getItems(FormApp.ItemType.GRID).length + ' (harus 50)');
  Logger.log('Link untuk penilai : ' + form.getPublishedUrl());
  Logger.log('Link edit (untuk Anda): ' + form.getEditUrl());
}

function aturForm_(form) {
  form.setTitle(JUDUL)
      .setDescription(PENGANTAR)
      .setCollectEmail(false)
      .setAllowResponseEdits(true)
      .setProgressBar(true)
      .setShowLinkToRespondAgain(false);
}

/* ---------------------------------------------------------------- membuat form */
function buatForm() {
  const soal = bacaSoal_();
  const form = FormApp.create(JUDUL);
  Logger.log('Form dibuat: ' + form.getEditUrl() + '  (pakai link ini untuk lengkapiForm jika berhenti)');
  aturForm_(form);
  form.addTextItem().setTitle('Nama penilai').setRequired(true);
  hubungkanSheet_(form);
  soal.forEach(s => tambahSoal_(form, s, soal.length));
  lapor_(form);
}

function lengkapiForm() {
  if (!FORM_URL) throw new Error('Isi FORM_URL di bagian atas dengan link edit form, lalu simpan.');
  const form = FormApp.openByUrl(FORM_URL);
  const soal = bacaSoal_();
  if (!form.getDescription()) aturForm_(form);          // form kosong yang dibuat manual
  if (!form.getItems(FormApp.ItemType.TEXT).some(it => it.getTitle() === 'Nama penilai')) {
    form.moveItem(form.addTextItem().setTitle('Nama penilai').setRequired(true), 0);
  }
  const judul = new Set(form.getItems().map(it => it.getTitle()));
  const lengkap = s => judul.has(judulHalaman_(s, soal.length)) && judul.has(judulSoal_(s)) &&
                       s.jawaban.every(j => judul.has(j.id) && judul.has(judulCatatan_(j.id)));
  const mulai = soal.findIndex(s => !lengkap(s));
  if (mulai < 0) { Logger.log('Semua soal sudah lengkap.'); lapor_(form); return; }

  // hapus sisa soal yang terpotong (soal ke-mulai dan sesudahnya), lalu tambahkan ulang
  const sisa = new Set();
  soal.slice(mulai).forEach(s => {
    sisa.add(judulHalaman_(s, soal.length)); sisa.add(judulSoal_(s));
    s.jawaban.forEach(j => { sisa.add(j.id); sisa.add(judulCatatan_(j.id)); });
  });
  form.getItems().filter(it => sisa.has(it.getTitle())).forEach(it => form.deleteItem(it));
  soal.slice(mulai).forEach(s => tambahSoal_(form, s, soal.length));
  hubungkanSheet_(form);
  Logger.log('Soal ditambahkan mulai soal ' + soal[mulai].no);
  lapor_(form);
}

/* ------------------------------------------------------------------------------------------
 * Setelah penilai mengirim: salin jawaban terakhir ke sheet Penilaian (skor kolom G-J, catatan L).
 * ------------------------------------------------------------------------------------------ */
function tabJawaban_(ss) {
  if (!FORM_URL) throw new Error('Isi FORM_URL dengan link edit form final, lalu simpan.');
  const formId = FormApp.openByUrl(FORM_URL).getId();
  const tab = ss.getSheets().find(sh => {
    const u = sh.getFormUrl();
    return u && u.indexOf(formId) !== -1;
  });
  if (!tab) throw new Error('Tidak ada tab di spreadsheet ini yang tersambung ke form tersebut. ' +
                            'Buka form > Responses > Link to Sheets, pilih spreadsheet ini.');
  return tab;
}

function salinKePenilaian() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const resp = tabJawaban_(ss);
  const sheet = ss.getSheetByName(SHEET_NAME);
  const data = resp.getDataRange().getValues();
  if (data.length < 2) throw new Error('Belum ada jawaban di tab ' + resp.getName());
  const isiRows = data.slice(1).filter(r => String(r[0]).trim());   // abaikan baris kosong
  if (!isiRows.length) throw new Error('Belum ada jawaban di tab ' + resp.getName());
  const head = data[0], row = isiRows[isiRows.length - 1];      // jawaban terakhir
  const ids = sheet.getRange(2, 2, sheet.getLastRow() - 1, 1).getValues().map(r => String(r[0]));
  const skor = ids.map(() => ['', '', '', '']);
  const cat = ids.map(() => ['']);
  head.forEach((h, c) => {
    h = String(h);
    const m = h.match(/^(\d{2}-[AB]) \[(.+)\]$/);              // "01-A [Akurasi]"
    if (m) {
      const i = ids.indexOf(m[1]), k = KRITERIA.indexOf(m[2]);
      const v = parseInt(String(row[c]), 10);                   // "5 (Sangat baik)" -> 5
      if (i >= 0 && k >= 0 && !isNaN(v)) skor[i][k] = v;
      return;
    }
    const mc = h.match(/^Catatan untuk (\d{2}-[AB]) /);
    if (mc && String(row[c]).trim()) {
      const i = ids.indexOf(mc[1]);
      if (i >= 0) cat[i][0] = String(row[c]).trim();
    }
  });
  sheet.getRange(2, 7, ids.length, 4).setValues(skor);   // kolom G-J
  sheet.getRange(2, 12, ids.length, 1).setValues(cat);   // kolom L
  const lengkap = skor.filter(s => s.every(v => v !== '')).length;
  Logger.log('Disalin dari jawaban ' + row[0] + ' (' + row[1] + '): ' + lengkap + ' dari ' + ids.length + ' jawaban lengkap');
}

/* ------------------------------------------------------------------------------------------
 * Uji coba tanpa mengisi 50 jawaban secara manual (butuh FORM_URL terisi):
 *   isiUjiCoba       -> kirim 1 jawaban palsu atas nama "UJI COBA". Jawaban ke-i (01-A = 0, 01-B = 1, ...),
 *                       kriteria ke-k: skor ((i + k) % 5) + 1, jadi 01-A = 1,2,3,4 dan 01-B = 2,3,4,5.
 *                       Catatan uji coba ditulis untuk 02-A.
 *   salinKePenilaian -> cek baris 01-A di sheet Penilaian harus berisi 1, 2, 3, 4.
 *   hapusUjiCoba     -> hapus jawaban uji coba. Fungsi ini MENOLAK berjalan jika ada jawaban dari
 *                       penilai sungguhan (nama selain "UJI COBA"), sehingga jawaban penilai aman.
 * ------------------------------------------------------------------------------------------ */
function isiUjiCoba() {
  if (!FORM_URL) throw new Error('Isi FORM_URL dengan link edit form final, lalu simpan.');
  const form = FormApp.openByUrl(FORM_URL);
  const resp = form.createResponse();
  let i = 0;
  form.getItems().forEach(it => {
    const tipe = it.getType();
    if (tipe === FormApp.ItemType.TEXT) {
      resp.withItemResponse(it.asTextItem().createResponse(NAMA_UJI));
    } else if (tipe === FormApp.ItemType.GRID) {
      resp.withItemResponse(it.asGridItem().createResponse(KRITERIA.map((_, k) => SKOR[(i + k) % 5])));
      i++;
    } else if (tipe === FormApp.ItemType.PARAGRAPH_TEXT && it.getTitle() === judulCatatan_('02-A')) {
      resp.withItemResponse(it.asParagraphTextItem().createResponse('catatan uji coba'));
    }
  });
  resp.submit();
  Logger.log('Jawaban uji coba terkirim. Jalankan salinKePenilaian lalu cek sheet Penilaian.');
}

function hapusUjiCoba() {
  if (!FORM_URL) throw new Error('Isi FORM_URL dengan link edit form final, lalu simpan.');
  const form = FormApp.openByUrl(FORM_URL);
  const namaItem = form.getItems(FormApp.ItemType.TEXT).find(it => it.getTitle() === 'Nama penilai');
  const asli = form.getResponses().filter(r => {
    const ir = namaItem ? r.getResponseForItem(namaItem) : null;
    return !ir || String(ir.getResponse()).trim() !== NAMA_UJI;
  });
  if (asli.length) throw new Error('DIBATALKAN: ada ' + asli.length + ' jawaban dari penilai sungguhan. ' +
                                   'Tidak ada yang dihapus.');
  form.deleteAllResponses();
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const resp = tabJawaban_(ss);
  if (resp.getLastRow() > 1) resp.getRange(2, 1, resp.getLastRow() - 1, resp.getLastColumn()).clearContent();
  const sheet = ss.getSheetByName(SHEET_NAME);
  sheet.getRange(2, 7, sheet.getLastRow() - 1, 4).clearContent();   // G-J
  sheet.getRange(2, 12, sheet.getLastRow() - 1, 1).clearContent();  // L
  Logger.log('Jawaban uji coba dihapus dari form, tab ' + resp.getName() + ', dan sheet Penilaian.');
}
