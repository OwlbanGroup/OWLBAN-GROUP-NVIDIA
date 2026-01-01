import {
  Entity,
  PrimaryGeneratedColumn,
  Column,
  OneToMany,
  CreateDateColumn,
} from 'typeorm';
import { User } from '../users/user.entity';
import { BankConnection } from '../bank-connections/bank-connection.entity';

export type OrganizationType = 'PERSONAL' | 'BUSINESS' | 'CORPORATE';

@Entity('organizations')
export class Organization {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column()
  name: string;

  @Column({ type: 'varchar' })
  type: OrganizationType;

  @OneToMany(() => User, (user) => user.organization)
  users: User[];

  @OneToMany(() => BankConnection, (bc) => bc.organization)
  bankConnections: BankConnection[];

  @CreateDateColumn()
  createdAt: Date;
}
