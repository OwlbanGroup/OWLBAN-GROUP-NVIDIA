import { Entity, Column, ManyToOne, JoinColumn, Index, PrimaryGeneratedColumn, CreateDateColumn, UpdateDateColumn } from 'typeorm';
import { Payment } from '../../payments-core/entities/payment.entity';

export enum AchSecCode {
  PPD = 'PPD', // Prearranged Payment and Deposit
  CCD = 'CCD', // Corporate Credit or Debit
  WEB = 'WEB', // Internet-Initiated Entry
  TEL = 'TEL', // Telephone-Initiated Entry
  CTX = 'CTX', // Corporate Trade Exchange
}

export enum AchTransactionType {
  CREDIT = 'CREDIT',
  DEBIT = 'DEBIT',
}

@Entity('ach_payments')
@Index(['paymentId'])
@Index(['batchId'])
@Index(['status'])
export class AchPayment {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  // Link to base payment
  @ManyToOne(() => Payment, { nullable: false, onDelete: 'CASCADE' })
  @JoinColumn({ name: 'paymentId' })
  payment: Payment;

  @Column({ nullable: false })
  paymentId: string;

  // ACH-specific fields
  @Column({ type: 'enum', enum: AchSecCode, nullable: false })
  secCode: AchSecCode;

  @Column({ type: 'enum', enum: AchTransactionType, nullable: false })
  transactionType: AchTransactionType;

  @Column({ nullable: false })
  originatorName: string;

  @Column({ nullable: false })
  originatorId: string;

  @Column({ nullable: false })
  receiverName: string;

  @Column({ nullable: false })
  receiverAccountNumber: string;

  @Column({ nullable: false })
  receiverRoutingNumber: string;

  @Column({ type: 'text', nullable: true })
  addendaRecord: string;

  @Column({ default: false })
  sameDayAch: boolean;

  @Column({ type: 'date', nullable: true })
  effectiveDate: Date;

  @Column({ nullable: true })
  batchId: string;

  @Column({ nullable: true })
  traceNumber: string;

  @Column({ nullable: true })
  returnCode: string;

  @Column({ type: 'text', nullable: true })
  returnReason: string;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;
}
