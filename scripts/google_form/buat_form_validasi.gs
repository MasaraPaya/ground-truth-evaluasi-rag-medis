/**
 * Membuat Google Form validasi dataset QA dari sheet "Validasi"
 * (Google Sheets hasil konversi Lembar_Validasi_Dataset_QA.xlsx).
 *
 * Setiap QA = 1 kotak grid berjudul nomor urut (teks QA + 3 kriteria x skor 1-5).
 * Kolom catatan opsional: di bawah setiap QA (CATATAN_PER_QA = true) atau satu per halaman (false).
 * Nomor = urutan di dataset (1 = DDR-001, ..., 100 = XDM-015); padanannya ada di kolom ID sheet Validasi.
 *
 * Cara pakai (Extensions > Apps Script dari spreadsheet itu):
 *   1. Pilih fungsi buatForm > Run (BUKAN Debug).
 *   2. Jika berhenti karena batas waktu, isi FORM_URL dengan link edit form yang terbentuk,
 *      lalu jalankan lengkapiForm. Boleh diulang sampai log menulis "total QA di form: 100".
 */

const SHEET_NAME = 'Validasi';
const QA_PER_PAGE = 10;
const MAX_DESC = 3500;
const CATATAN_PER_QA = true;  // true = kolom catatan di bawah setiap QA; false = satu kolom per halaman
const FORM_URL = '';  // isi hanya untuk lengkapiForm, contoh: 'https://docs.google.com/forms/d/.../edit'

const KRITERIA = ['Kejelasan pertanyaan', 'Kebahasaan', 'Kesesuaian jawaban dengan sumber'];
const SKOR = ['1 (Sangat kurang)', '2', '3', '4', '5 (Sangat baik)'];

const PENGANTAR =
  '"Pengembangan Ground-Truth Dataset Penyakit Fisiologis dan Evaluasi Akurasi Penalaran GraphRAG ' +
  'Menggunakan Metrik RAGAS" \n\nRauf Hafizh Asmenta\n202210370311209\n\n' +
  'Setiap pasangan pertanyaan-jawaban (QA) dinilai pada tiga kriteria dengan skala 1-5:\n' +
  '- Kejelasan pertanyaan: 5 = sangat jelas dan spesifik ... 1 = tidak dapat dipahami\n' +
  '- Kebahasaan: 5 = tata bahasa dan istilah biomedis tepat ... 1 = tidak dapat dipahami\n' +
  '- Kesesuaian jawaban dengan sumber: 5 = sepenuhnya sesuai konteks/sumber ... 1 = tidak sesuai\n\n' +
  'Catatan:\n' +
  '- Sumber PrimeKG: kesesuaian dinilai terhadap fakta PrimeKG di bagian Konteks. Jika jawaban sesuai ' +
  'PrimeKG tetapi janggal secara klinis, mohon tulis di kolom komentar.\n' +
  '- Sumber MedQA: soal ujian USMLE yang dipakai apa adanya (verbatim), termasuk ejaan aslinya.\n' +
  '- Sumber Harrison: soal konseptual dari Harrison\'s Principles of Internal Medicine ed. 20.\n' +
  '- Konteks ditulis persis seperti di PrimeKG (kepala --[relasi]--> ekor) dan tidak perlu dinilai.\n' +
  '- Jika Ibu login ke akun Google, isian tersimpan otomatis sehingga dapat dilanjutkan nanti.';

function bacaQA_() {
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(SHEET_NAME);
  if (!sheet) throw new Error('Sheet "' + SHEET_NAME + '" tidak ditemukan.');
  return sheet.getDataRange().getValues().slice(1).filter(r => r[0]);
}

function judulHalaman_(i, n) {
  return 'QA ' + (i + 1) + '-' + Math.min(i + QA_PER_PAGE, n) + ' dari ' + n;
}

function komentarJudul_(i, n) {
  return 'Komentar untuk QA ' + (Math.floor(i / QA_PER_PAGE) * QA_PER_PAGE + 1) + '-' +
         Math.min(Math.floor(i / QA_PER_PAGE) * QA_PER_PAGE + QA_PER_PAGE, n) + ' (opsional)';
}

function tambahKomentar_(form, i, n) {
  form.addParagraphTextItem()
      .setTitle(komentarJudul_(i, n))
      .setHelpText('Tulis nomor QA lalu komentarnya, satu baris per QA. Contoh:\n3: istilah "insulin human" kurang lazim\n7: jawaban kurang lengkap');
}

function catatanQA_(no) { return 'Catatan untuk QA ' + no + ' (opsional)'; }

function akhirHalaman_(i, n) { return i % QA_PER_PAGE === QA_PER_PAGE - 1 || i === n - 1; }

function tambahQA_(form, r, no) {
  const [id, kategori, sumber, tingkat, pertanyaan, jawaban, konteks, catatan] = r;
  let desc = 'Kategori: ' + kategori + ' | Sumber: ' + sumber + ' | Tingkat: ' + tingkat +
             (catatan ? ' | ' + catatan : '') +
             '\n\nPertanyaan: ' + pertanyaan +
             '\n\nJawaban: ' + jawaban +
             '\n\nKonteks:\n' + konteks;
  if (desc.length > MAX_DESC) desc = desc.slice(0, MAX_DESC) + '\n... (konteks lengkap ada di lembar Excel)';
  form.addGridItem()
      .setTitle(String(no))
      .setHelpText(desc)
      .setRows(KRITERIA)
      .setColumns(SKOR)
      .setRequired(true);   // satu skor wajib untuk setiap kriteria
  if (CATATAN_PER_QA) form.addParagraphTextItem().setTitle(catatanQA_(no));
}

function hubungkanSheet_(form) {
  try { form.getDestinationId(); } catch (e) {
    form.setDestination(FormApp.DestinationType.SPREADSHEET, SpreadsheetApp.getActiveSpreadsheet().getId());
  }
}

function lapor_(form) {
  const total = form.getItems(FormApp.ItemType.GRID).length;
  Logger.log('Total QA di form: ' + total);
  Logger.log('Link untuk validator : ' + form.getPublishedUrl());
  Logger.log('Link edit (untuk Anda): ' + form.getEditUrl());
}

function buatForm() {
  const rows = bacaQA_();
  const form = FormApp.create('Validasi Ahli Dataset Ground-Truth QA GraphRAG');
  Logger.log('Form dibuat: ' + form.getEditUrl() + '  (pakai link ini untuk lengkapiForm jika berhenti)');
  form.setDescription(PENGANTAR)
      .setCollectEmail(false)
      .setAllowResponseEdits(true)
      .setProgressBar(true)
      .setShowLinkToRespondAgain(false);
  form.addTextItem().setTitle('Nama validator').setRequired(true);
  hubungkanSheet_(form);

  rows.forEach((r, i) => {
    if (i % QA_PER_PAGE === 0) form.addPageBreakItem().setTitle(judulHalaman_(i, rows.length));
    tambahQA_(form, r, i + 1);
    if (!CATATAN_PER_QA && akhirHalaman_(i, rows.length)) tambahKomentar_(form, i, rows.length);
  });
  lapor_(form);
}

function lengkapiForm() {
  if (!FORM_URL) throw new Error('Isi FORM_URL di bagian atas dengan link edit form, lalu simpan.');
  const form = FormApp.openByUrl(FORM_URL);
  const rows = bacaQA_();

  // form kosong yang dibuat manual: pasang pengantar, pengaturan, dan isian nama validator
  if (!form.getDescription()) {
    form.setTitle('Validasi Ahli Dataset Ground-Truth QA GraphRAG')
        .setDescription(PENGANTAR)
        .setCollectEmail(false)
        .setAllowResponseEdits(true)
        .setProgressBar(true)
        .setShowLinkToRespondAgain(false);
  }
  if (!form.getItems(FormApp.ItemType.TEXT).some(it => it.getTitle() === 'Nama validator')) {
    form.moveItem(form.addTextItem().setTitle('Nama validator').setRequired(true), 0);
  }

  const ada = new Set(), komentar = new Set(), halaman = new Set();
  form.getItems().forEach(it => {
    const t = it.getTitle(), tipe = it.getType();
    if (tipe === FormApp.ItemType.GRID) ada.add(t);
    if (tipe === FormApp.ItemType.PARAGRAPH_TEXT) komentar.add(t);
    if (tipe === FormApp.ItemType.PAGE_BREAK) halaman.add(t);
  });
  // QA yang belum ada ditambahkan; kolom komentar halaman yang belum sempat dibuat ditambahkan di akhir halamannya
  let added = 0;
  rows.forEach((r, i) => {
    const id = String(i + 1);
    if (i % QA_PER_PAGE === 0 && !ada.has(id) && !halaman.has(judulHalaman_(i, rows.length))) {
      form.addPageBreakItem().setTitle(judulHalaman_(i, rows.length));
    }
    if (!ada.has(id)) { tambahQA_(form, r, i + 1); added++; }
    else if (CATATAN_PER_QA && !komentar.has(catatanQA_(id))) {   // grid ada, catatannya terpotong
      const gi = form.getItems().findIndex(x => x.getType() === FormApp.ItemType.GRID && x.getTitle() === id);
      form.moveItem(form.addParagraphTextItem().setTitle(catatanQA_(id)), gi + 1);
    }
    if (!CATATAN_PER_QA && akhirHalaman_(i, rows.length) && !komentar.has(komentarJudul_(i, rows.length))) {
      tambahKomentar_(form, i, rows.length);
    }
  });
  hubungkanSheet_(form);
  Logger.log('QA ditambahkan: ' + added);
  lapor_(form);
}

/* ------------------------------------------------------------------------------------------
 * Setelah validator mengirim: salin jawaban terakhir ke sheet Validasi (skor kolom I-K, komentar L),
 * sehingga sheet Rekap langsung terhitung. Tab jawaban dicari otomatis dari FORM_URL.
 * ------------------------------------------------------------------------------------------ */
const RESP_SHEET = '';  // kosongkan: tab jawaban dicari otomatis dari FORM_URL

function tabJawaban_(ss) {
  if (RESP_SHEET) return ss.getSheetByName(RESP_SHEET);
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

function salinKeValidasi() {
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const resp = tabJawaban_(ss);
  if (!resp) throw new Error('Tab "' + RESP_SHEET + '" tidak ditemukan. Sesuaikan RESP_SHEET.');
  const val = ss.getSheetByName(SHEET_NAME);
  const data = resp.getDataRange().getValues();
  if (data.length < 2) throw new Error('Belum ada jawaban di tab ' + resp.getName());
  const head = data[0], row = data[data.length - 1];          // jawaban terakhir
  const n = val.getLastRow() - 1;
  const skor = Array.from({length: n}, () => ['', '', '']);
  const kom = Array.from({length: n}, () => ['']);
  const tambahKom = (no, teks) => {
    if (no >= 1 && no <= n && teks) kom[no - 1][0] += (kom[no - 1][0] ? ' | ' : '') + teks;
  };

  head.forEach((h, c) => {
    h = String(h);
    const m = h.match(/^(\d+) \[(.+)\]$/);                   // "12 [Kebahasaan]"
    if (m) {
      const k = KRITERIA.indexOf(m[2]);
      const v = parseInt(String(row[c]), 10);                   // "5 (Sangat baik)" -> 5
      if (k >= 0 && !isNaN(v)) skor[Number(m[1]) - 1][k] = v;
      return;
    }
    const isi = String(row[c]).trim();
    if (!isi) return;
    const satu = h.match(/^Catatan untuk QA (\d+) /);           // catatan per QA
    if (satu) { tambahKom(Number(satu[1]), isi); return; }
    const per = h.match(/^Komentar untuk QA (\d+)-(\d+)/);    // komentar per halaman: "3: ..."
    if (per) {
      isi.split('\n').forEach(line => {
        const mm = line.match(/^\s*(\d+)\s*[:.\-]\s*(.+)$/);
        if (mm) tambahKom(Number(mm[1]), mm[2].trim());
        else tambahKom(Number(per[1]), '(QA ' + per[1] + '-' + per[2] + ') ' + line.trim());
      });
    }
  });
  val.getRange(2, 9, n, 3).setValues(skor);    // kolom I-K
  val.getRange(2, 12, n, 1).setValues(kom);    // kolom L
  Logger.log('Disalin dari jawaban ' + row[0] + ' (' + row[1] + ')');
}

/* ------------------------------------------------------------------------------------------
 * Uji coba tanpa mengisi 100 QA secara manual (butuh FORM_URL terisi):
 *   isiUjiCoba     -> kirim 1 jawaban palsu lengkap. Skor QA ke-n, kriteria ke-k = ((n + k) % 5) + 1,
 *                     jadi QA 1 = 2, 3, 4 dan QA 2 = 3, 4, 5. Komentar uji coba ditulis untuk QA 3.
 *   salinKeValidasi -> cek: baris DDR-001 di sheet Validasi harus berisi 2, 3, 4.
 *   hapusUjiCoba   -> hapus semua jawaban uji coba sebelum link dikirim ke validator.
 * ------------------------------------------------------------------------------------------ */
function isiUjiCoba() {
  if (!FORM_URL) throw new Error('Isi FORM_URL dengan link edit form final, lalu simpan.');
  const form = FormApp.openByUrl(FORM_URL);
  const resp = form.createResponse();
  form.getItems().forEach(it => {
    const tipe = it.getType();
    if (tipe === FormApp.ItemType.TEXT) {
      resp.withItemResponse(it.asTextItem().createResponse('UJI COBA'));
    } else if (tipe === FormApp.ItemType.GRID) {
      const no = Number(it.getTitle());
      resp.withItemResponse(it.asGridItem().createResponse(KRITERIA.map((_, k) => SKOR[(no + k) % 5])));
    } else if (tipe === FormApp.ItemType.PARAGRAPH_TEXT && it.getTitle().startsWith('Komentar untuk QA 1-')) {
      resp.withItemResponse(it.asParagraphTextItem().createResponse('3: komentar uji coba'));
    } else if (tipe === FormApp.ItemType.PARAGRAPH_TEXT && it.getTitle() === catatanQA_(3)) {
      resp.withItemResponse(it.asParagraphTextItem().createResponse('komentar uji coba'));
    }
  });
  resp.submit();
  Logger.log('Jawaban uji coba terkirim. Jalankan salinKeValidasi lalu cek sheet Validasi.');
}

function hapusUjiCoba() {
  if (!FORM_URL) throw new Error('Isi FORM_URL dengan link edit form final, lalu simpan.');
  FormApp.openByUrl(FORM_URL).deleteAllResponses();
  const ss = SpreadsheetApp.getActiveSpreadsheet();
  const resp = tabJawaban_(ss);
  if (resp && resp.getLastRow() > 1) {
    resp.getRange(2, 1, resp.getLastRow() - 1, resp.getLastColumn()).clearContent();
  }
  const val = ss.getSheetByName(SHEET_NAME);
  val.getRange(2, 9, val.getLastRow() - 1, 4).clearContent();   // kolom I-L (format & validasi tetap)
  Logger.log('Jawaban uji coba dihapus dari form, tab ' + resp.getName() + ', dan sheet Validasi.');
}
