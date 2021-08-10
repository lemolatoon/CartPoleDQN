# カスタム進捗表示用のCallback関数定義
"""
進捗表示用のCallback関数です。
Batch終了時とEpoch終了時にデータを収集して、表示しています。
ポイントとしては print 出力時に /r で行先頭にカーソルを戻しつつ、引数 end='' で改行を抑制している点です。
"""
import datetime
import tensorflow as tf

class DisplayCallBack(tf.keras.callbacks.Callback):
  # コンストラクタ
  def __init__(self):
    self.last_acc, self.last_loss, self.last_val_acc, self.last_val_loss = None, None, None, None
    self.now_batch, self.now_epoch = None, None

    self.epochs, self.samples, self.batch_size = None, None, None

  # カスタム進捗表示 (表示部本体)
  def print_progress(self):
    epoch = self.now_epoch
    batch = self.now_batch

    epochs = self.epochs
    samples = self.samples
    batch_size = self.batch_size
    sample = batch_size*(batch)

    # '\r' と end='' を使って改行しないようにする
    if self.last_val_acc and self.last_val_loss:
      # val_acc/val_loss が表示可能
      print("\rEpoch %d/%d (%d/%d) -- acc: %f loss: %f - val_acc: %f val_loss: %f" % (epoch+1, epochs, sample, samples, self.last_acc, self.last_loss, self.last_val_acc, self.last_val_loss), end='')
    else:
      # val_acc/val_loss が表示不可
      print("\rEpoch %d/%d (%d/%d) -- acc: %f loss: %f" % (epoch+1, epochs, sample, samples, self.last_acc, self.last_loss), end='')


  # fit開始時
  def on_train_begin(self, logs={}):
    print('\n##### Train Start ##### ' + str(datetime.datetime.now()))

    # パラメータの取得
    self.epochs = self.params['epochs']
    self.samples = self.params['samples']
    self.batch_size = self.params['batch_size']

    # 標準の進捗表示をしないようにする
    self.params['verbose'] = 0


  # batch開始時
  def on_batch_begin(self, batch, logs={}):
    self.now_batch = batch

  # batch完了時 (進捗表示)
  def on_batch_end(self, batch, logs={}):
    # 最新情報の更新
    self.last_acc = logs.get('acc') if logs.get('acc') else 0.0
    self.last_loss = logs.get('loss') if logs.get('loss') else 0.0

    # 進捗表示
    self.print_progress()


  # epoch開始時
  def on_epoch_begin(self, epoch, log={}):
    self.now_epoch = epoch

  # epoch完了時 (進捗表示)
  def on_epoch_end(self, epoch, logs={}):
    # 最新情報の更新
    self.last_val_acc = logs.get('val_acc') if logs.get('val_acc') else 0.0
    self.last_val_loss = logs.get('val_loss') if logs.get('val_loss') else 0.0

    # 進捗表示
    self.print_progress()


  # fit完了時
  def on_train_end(self, logs={}):
    print('\n##### Train Complete ##### ' + str(datetime.datetime.now()))